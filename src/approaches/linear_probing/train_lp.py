"""
train_lp.py — Main execution script for the Linear Probing (LP) run.

Supports two modes:
  - Official mode (--official): frozen REVE encoder runs live each batch,
    trainable cls_query_token + linear head.
  - Fast mode (--fast): pre-computed REVE embeddings + standalone linear head.
    Quick iteration, not a faithful reproduction.

Supports two datasets:
  - FACED (123 subjects, 32 channels)
  - THU-EP (79 subjects, 30 channels)

Run with:
    # Official mode, FACED, 9-class, all folds
    uv run python -m src.approaches.linear_probing.train_lp --dataset faced --task 9-class

    # Official mode, THU-EP, binary, single fold
    uv run python -m src.approaches.linear_probing.train_lp --dataset thu-ep --task binary --fold 1

    # Fast mode (pre-computed embeddings)
    uv run python -m src.approaches.linear_probing.train_lp --dataset thu-ep --fast --task binary

    # Generalization mode
    uv run python -m src.approaches.linear_probing.train_lp --dataset faced --task binary --generalization

    CUDA_VISIBLE_DEVICES=1 uv run python -m src.approaches.linear_probing.train_lp --dataset faced --task 9-class --pooling last 
"""

from __future__ import annotations

import argparse
import gc
import random
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
from lightning.pytorch.loggers import WandbLogger, CSVLogger
import wandb

from src.datasets.folds import (
    get_all_subjects, get_kfold_splits, get_stimulus_generalization_split, N_FOLDS,
)

from src.approaches.linear_probing.model import (
    ReveClassifierLP, EmbeddingExtractor, LinearProber, evaluate_model,
)
from src.approaches.shared.reve import load_reve_and_positions, get_channel_names
from src.approaches.linear_probing.config import (
    LPConfig, OUTPUT_DIR,
    USE_WANDB, WANDB_PROJECT, WANDB_ENTITY,
    SAMPLING_RATE, DEVICE, ACCELERATOR, NUM_WORKERS,
)
from src.approaches.shared.stable_adamw import StableAdamW
from src.approaches.shared.training_utils import fmt_dur, COL_W, _get_exponential_warmup_lambda
from src.approaches.shared.dataset import build_raw_dataset
from src.approaches.linear_probing.summary import (
    print_fold_summary, print_cross_seed_summary,
)


def fmt_metric(val: float, decimals: int = 4) -> str:
    import math
    if math.isnan(val):
        return "n/a"
    return f"{val:.{decimals}f}"


# ── Patience monitor ─────────────────────────────────────

class PatienceMonitor:
    """Monitor validation accuracy with early stopping."""

    def __init__(self, patience: int = 10):
        self.patience = patience
        self.best_acc = 0.0
        self.counter = 0

    def __call__(self, val_acc: float) -> bool:
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


# ── Official-mode training loop ──────────────────────────────────────────────

def train_official_mode(
    cfg: LPConfig,
    model: ReveClassifierLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
) -> dict:
    """
    Manual training loop matching REVE linear probing procedure.

    Uses: StableAdamW, exponential warmup, mixup, AMP, gradient clipping.
    """
    model.to(device)
    device_type = "cuda" if "cuda" in device else "cpu"

    # Optimizer: only trainable params (cls_query_token + linear_head)
    optimizer = StableAdamW(
        model.trainable_parameters(),
        lr=cfg.lr,
        betas=(0.92, 0.999),
        weight_decay=cfg.weight_decay,
    )

    # Warmup scheduler (step-level, exponential)
    warmup_steps = cfg.warmup_epochs * len(train_loader)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_get_exponential_warmup_lambda(warmup_steps),
    )

    # ReduceLROnPlateau (epoch-level, on validation accuracy)
    reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=cfg.scheduler_patience,
    )

    scaler = torch.amp.GradScaler()
    patience_monitor = PatienceMonitor(cfg.early_stop_patience)

    best_metrics = {}
    best_acc = 0.0
    best_state = None

    print(f"\n{'─' * COL_W}")
    header = (
        f"{'Epoch':>6}  {'EpTime':>7}  {'Elapsed':>8}  "
        f"{'TrLoss':>8}  {'TrAcc':>7}  {'VaLoss':>8}  {'VaAcc':>7}  {'VaBalAcc':>9}  "
        f"{'VaAUROC':>8}  {'VaF1w':>7}  {'LR':>10}"
    )
    print(header)
    print(f"{'─' * COL_W}")

    fit_start = time.time()

    for epoch in range(cfg.max_epochs):
        epoch_start = time.time()
        # Official uses epoch <= warmup_epochs (inclusive)
        warmup_active = epoch <= cfg.warmup_epochs

        # ── Train ──────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        n_correct = 0
        n_samples = 0

        for batch in train_loader:
            eeg, target = batch[0].to(device), batch[1].long().to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device_type, enabled=cfg.use_amp, dtype=torch.float16):
                if cfg.use_mixup:
                    mm = random.random()
                    perm = torch.randperm(eeg.size(0), device=device)
                    output = model(mm * eeg + (1 - mm) * eeg[perm])
                    loss = mm * F.cross_entropy(output, target) + (1 - mm) * F.cross_entropy(output, target[perm])
                else:
                    output = model(eeg)
                    loss = F.cross_entropy(output, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()

            # Step-level warmup (only if scaler didn't skip)
            if warmup_active and scale_before == scaler.get_scale():
                warmup_scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            # Accuracy against original (unshuffled) labels
            with torch.no_grad():
                preds = output.argmax(dim=1)
            n_correct += (preds == target).sum().item()
            n_samples += target.size(0)

        avg_loss = epoch_loss / max(n_batches, 1)
        train_acc = n_correct / max(n_samples, 1)

        # ── Validate ───────────────────────────────────────────────────
        metrics = evaluate_model(
            model, val_loader, device=device,
            n_classes=cfg.num_classes, use_amp=cfg.use_amp,
        )

        val_acc = metrics["accuracy"]
        reduce_scheduler.step(val_acc)

        # Track best
        if val_acc > best_acc:
            best_acc = val_acc
            best_metrics = {**metrics, "epoch": epoch + 1}
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()
                          if k.startswith("cls_query_token") or k.startswith("linear_head")}

        # Print epoch summary
        ep_time = time.time() - epoch_start
        elapsed = time.time() - fit_start
        lr = optimizer.param_groups[0]["lr"]
        avg_ep = elapsed / (epoch + 1)
        remaining = avg_ep * (cfg.max_epochs - epoch - 1)
        eta = f"ETA {fmt_dur(remaining)}" if epoch + 1 < cfg.max_epochs else "done"

        print(
            f"{epoch+1:>6}  {fmt_dur(ep_time):>7}  {fmt_dur(elapsed):>8}  "
            f"{avg_loss:>8.4f}  {train_acc:>7.4f}  {metrics['val_loss']:>8.4f}  {val_acc:>7.4f}  {metrics['balanced_acc']:>9.4f}  "
            f"{metrics['auroc']:>8.4f}  {metrics['f1_weighted']:>7.4f}  "
            f"{lr:>10.2e}  ({eta})"
        )

        # W&B per-epoch logging
        if wandb.run is not None:
            wandb.log({
                "train/loss":    avg_loss,
                "train/acc":     train_acc,
                "val/loss":      metrics["val_loss"],
                "val/acc":       val_acc,
                "val/bal_acc":   metrics["balanced_acc"],
                "val/auroc":     metrics["auroc"],
                "val/f1":        metrics["f1_weighted"],
                "train/lr":      lr,
            }, step=epoch + 1)

        # Early stopping
        if patience_monitor(val_acc):
            print(f"Early stopping at epoch {epoch + 1} (patience={cfg.early_stop_patience})")
            break

    total_time = time.time() - fit_start
    print(f"{'─' * COL_W}")
    print(
        f"Training complete — {epoch + 1} epochs | total: {fmt_dur(total_time)} | "
        f"best epoch={best_metrics.get('epoch', 'n/a')} val_acc={best_acc:.4f}"
    )
    print(f"{'─' * COL_W}")

    return {
        "val_acc":     best_metrics.get("accuracy"),
        "val_bal_acc": best_metrics.get("balanced_acc"),
        "val_auroc":   best_metrics.get("auroc"),
        "val_f1":      best_metrics.get("f1_weighted"),
        "best_epoch":  best_metrics.get("epoch"),
        "epochs_trained": epoch + 1,
        "best_state":  best_state,
    }


# ── Per-fold runner ──────────────────────────────────────────────────────────

def run_fold_official(
    cfg: LPConfig,
    fold_idx: int,
    train_subject_ids: list[int],
    val_subject_ids: list[int],
    reve_model: torch.nn.Module,
    pos_tensor: torch.Tensor,
    train_stimuli: set[int] | None = None,
    val_stimuli: set[int] | None = None,
    gen_seed: int | None = None,
) -> dict:
    """Run one fold in official mode."""
    seed_label = f"  |  gen_seed={gen_seed}" if gen_seed is not None else ""
    gen_tag = f"  |  GENERALIZATION{seed_label}" if cfg.generalization else ""

    print(f"\n{'#' * COL_W}")
    print(
        f"  Fold {fold_idx}/{N_FOLDS}  |  {cfg.dataset}  |  task={cfg.task_mode}  |  "
        f"pooling={cfg.pooling}  |  device={DEVICE}{gen_tag}"
    )
    print(
        f"  train: {len(train_subject_ids)} subjects  |  val: {len(val_subject_ids)} subjects"
    )
    print(f"{'#' * COL_W}")

    # Build raw datasets
    t_load = time.time()
    print("Building datasets ...", end="  ", flush=True)
    train_ds = build_raw_dataset(cfg, train_subject_ids, stimulus_filter=train_stimuli)
    val_ds   = build_raw_dataset(cfg, val_subject_ids, stimulus_filter=val_stimuli)
    print(
        f"done in {fmt_dur(time.time() - t_load)}  "
        f"(train={len(train_ds):,}  val={len(val_ds):,})"
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Build model
    model = ReveClassifierLP(
        reve_model=reve_model,
        pos_tensor=pos_tensor,
        n_classes=cfg.num_classes,
        n_channels=cfg.n_channels,
        window_size=cfg.window_size,
        pooling=cfg.pooling,
        dropout=cfg.dropout,
    )
    print(f"Trainable parameters: {model.n_trainable_params():,}")

    # torch.compile (PyTorch 2.0+) — speeds up repeated forward passes
    try:
        model = torch.compile(model)
        print("torch.compile: enabled")
    except Exception as e:
        print(f"torch.compile: skipped ({e})")

    # W&B logging
    run_name = cfg.run_name(fold_idx, gen_seed)
    if USE_WANDB:
        hparams = cfg.hparams_dict(
            fold_idx=fold_idx, n_folds=N_FOLDS,
            n_train_subjects=len(train_subject_ids),
            n_val_subjects=len(val_subject_ids),
            n_train_windows=len(train_ds),
            n_val_windows=len(val_ds),
            embed_dim=model._out_shape,
            gen_seed=gen_seed,
        )
        wandb.init(
            project=WANDB_PROJECT, entity=WANDB_ENTITY,
            name=run_name, group=cfg.group_name(),
            config=hparams, reinit=True,
        )

    # Train
    result = train_official_mode(cfg, model, train_loader, val_loader, DEVICE)

    # Save classifier weights
    ckpt_dir = OUTPUT_DIR / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if result.get("best_state"):
        weights_path = ckpt_dir / "classifier_weights.pt"
        torch.save(result["best_state"], weights_path)
        print(f"Classifier weights saved → {weights_path}")

    if USE_WANDB:
        wandb.log({
            "best/val_acc":     result.get("val_acc"),
            "best/val_bal_acc": result.get("val_bal_acc"),
            "best/val_auroc":   result.get("val_auroc"),
            "best/val_f1":      result.get("val_f1"),
        })
        wandb.finish()

    # Cleanup
    del model, train_ds, val_ds, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result["fold"] = fold_idx
    del result["best_state"]  # don't carry in summary
    return result


# ── Fast-mode helpers (pre-computed embeddings) ──────────────────────────────

class WarmupCallback(Callback):
    """Linear LR warmup over the first N epochs (fast mode)."""

    def __init__(self, warmup_epochs: int, base_lr: float) -> None:
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        epoch = trainer.current_epoch
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / self.warmup_epochs
            lr = self.base_lr * factor
            for pg in trainer.optimizers[0].param_groups:
                pg["lr"] = lr


def subject_cache_path(cfg: LPConfig, subject_id: int) -> Path:
    return (
        cfg.embeddings_dir / cfg.task_mode
        / f"ws{cfg.window_size}_st{cfg.stride}_{cfg.pool_tag}"
        / f"sub_{subject_id:02d}.pt"
    )


def precompute_all_subjects(
    cfg: LPConfig,
    all_subjects: list[int],
    reve_model: torch.nn.Module,
    pos_tensor: torch.Tensor,
) -> None:
    """Pre-compute REVE embeddings for every subject not yet cached."""
    missing = [sid for sid in all_subjects if not subject_cache_path(cfg, sid).exists()]
    if not missing:
        print(f"All {len(all_subjects)} subject embedding caches found.")
        return

    print(f"\n{len(missing)}/{len(all_subjects)} subjects need embedding extraction.")

    extractor = EmbeddingExtractor(reve_model, pos_tensor, device=DEVICE)

    t0 = time.time()
    for i, sid in enumerate(missing, 1):
        t_sub = time.time()
        print(f"\n[{i}/{len(missing)}] Subject {sid:02d}", end="  ")

        dataset = build_raw_dataset(
            LPConfig(
                dataset=cfg.dataset, task_mode=cfg.task_mode,
                window_size=cfg.window_size, stride=cfg.stride,
                scale_factor=cfg.scale_factor,
            ),
            [sid],
        )
        print(f"({len(dataset)} windows)", end="  ", flush=True)

        embeddings, labels, stim_indices = extractor.extract_embeddings(
            dataset, batch_size=cfg.batch_size,
            use_pooling=cfg.use_pooling, no_pool_mode=cfg.no_pool_mode,
        )
        EmbeddingExtractor.save_embeddings(
            embeddings, labels, subject_cache_path(cfg, sid),
            stimulus_indices=stim_indices,
        )
        print(f"done in {fmt_dur(time.time() - t_sub)}")
        del dataset, embeddings, labels
        gc.collect()

    print(f"\nAll subjects done. Total: {fmt_dur(time.time() - t0)}")


def load_subjects_embeddings(
    cfg: LPConfig,
    subject_ids: list[int],
    stimulus_filter: set[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_embs, all_labels = [], []
    for sid in subject_ids:
        path = subject_cache_path(cfg, sid)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        embs, labels = payload["embeddings"], payload["labels"]
        if stimulus_filter is not None:
            stim_idx = payload["stimulus_indices"]
            mask = torch.tensor([int(s.item()) in stimulus_filter for s in stim_idx], dtype=torch.bool)
            embs, labels = embs[mask], labels[mask]
        all_embs.append(embs)
        all_labels.append(labels)
    return torch.cat(all_embs, 0), torch.cat(all_labels, 0)


def run_fold_fast(
    cfg: LPConfig,
    fold_idx: int,
    train_subject_ids: list[int],
    val_subject_ids: list[int],
    train_stimuli: set[int] | None = None,
    val_stimuli: set[int] | None = None,
    gen_seed: int | None = None,
) -> dict:
    """Run one fold in fast mode (pre-computed embeddings + Lightning)."""
    from src.utils.callbacks import EpochSummaryCallback, fmt_dur as _fmt_dur

    seed_label = f"  |  gen_seed={gen_seed}" if gen_seed is not None else ""
    gen_tag = f"  |  GENERALIZATION{seed_label}" if cfg.generalization else ""

    print(f"\n{'#' * COL_W}")
    print(
        f"  Fold {fold_idx}/{N_FOLDS}  |  {cfg.dataset}  |  task={cfg.task_mode}  |  "
        f"FAST mode  |  device={DEVICE}{gen_tag}"
    )
    print(f"{'#' * COL_W}")

    t_load = time.time()
    print("Loading embeddings ...", end="  ", flush=True)
    train_embs, train_lbls = load_subjects_embeddings(cfg, train_subject_ids, stimulus_filter=train_stimuli)
    val_embs, val_lbls     = load_subjects_embeddings(cfg, val_subject_ids, stimulus_filter=val_stimuli)
    print(f"done in {_fmt_dur(time.time() - t_load)}  (train={train_embs.shape[0]:,}  val={val_embs.shape[0]:,})")

    train_loader = DataLoader(TensorDataset(train_embs, train_lbls), batch_size=cfg.batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(TensorDataset(val_embs, val_lbls), batch_size=cfg.batch_size, shuffle=False, num_workers=NUM_WORKERS)

    embed_dim = train_embs.shape[1]
    model = LinearProber(
        num_classes=cfg.num_classes, embed_dim=embed_dim, lr=cfg.lr,
        dropout=cfg.dropout, warmup_epochs=cfg.warmup_epochs,
        scheduler_patience=cfg.scheduler_patience,
        normalize_features=cfg.normalize_features,
    )

    run_name = cfg.run_name(fold_idx, gen_seed)
    ckpt_dir = OUTPUT_DIR / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    hparams = cfg.hparams_dict(
        fold_idx=fold_idx, n_folds=N_FOLDS,
        n_train_subjects=len(train_subject_ids),
        n_val_subjects=len(val_subject_ids),
        n_train_windows=int(train_embs.shape[0]),
        n_val_windows=int(val_embs.shape[0]),
        embed_dim=embed_dim, gen_seed=gen_seed,
    )

    if USE_WANDB:
        logger = WandbLogger(project=WANDB_PROJECT, entity=WANDB_ENTITY,
                             name=run_name, group=cfg.group_name(),
                             config=hparams, log_model=False, reinit=True)
    else:
        logger = CSVLogger(save_dir=str(ckpt_dir), name="csv_logs")

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir), filename="best-{epoch:02d}-{val/acc:.4f}",
        monitor="val/acc", mode="max", save_top_k=1, verbose=False,
    )
    early_stop_cb = EarlyStopping(monitor="val/acc", patience=cfg.early_stop_patience, mode="max", verbose=False)
    summary_cb = EpochSummaryCallback(
        output_dir=ckpt_dir, fold_idx=fold_idx, task_mode=cfg.task_mode,
        train_subjects=train_subject_ids, val_subjects=val_subject_ids, hparams=hparams,
    )
    warmup_cb = WarmupCallback(warmup_epochs=cfg.warmup_epochs, base_lr=cfg.lr)

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs, accelerator=ACCELERATOR, devices=1,
        logger=logger, callbacks=[checkpoint_cb, early_stop_cb, summary_cb, warmup_cb],
        log_every_n_steps=1, enable_progress_bar=False, enable_model_summary=True,
    )

    try:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    finally:
        del train_embs, train_lbls, val_embs, val_lbls, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if USE_WANDB:
            wandb.finish()

    # Save best weights
    best_ckpt = checkpoint_cb.best_model_path
    if best_ckpt:
        best_model = LinearProber.load_from_checkpoint(best_ckpt)
        torch.save(best_model.classifier.state_dict(), ckpt_dir / "classifier_weights.pt")
        del best_model

    valid_rows = [r for r in summary_cb.epoch_history if r["val_acc"] is not None]
    best_row = max(valid_rows, key=lambda r: r["val_acc"]) if valid_rows else {}
    return {
        "fold":           fold_idx,
        "val_acc":        best_row.get("val_acc"),
        "val_auroc":      best_row.get("val_auroc"),
        "val_f1":         best_row.get("val_f1"),
        "best_epoch":     best_row.get("epoch"),
        "epochs_trained": len(summary_cb.epoch_history),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> LPConfig:
    parser = argparse.ArgumentParser(description="Linear Probing with REVE")

    parser.add_argument("--dataset", choices=["faced", "thu-ep"], default="faced",
                        help="Dataset (default: faced)")
    parser.add_argument("--task", choices=["binary", "9-class"], default="binary",
                        help="Classification task (default: binary)")
    parser.add_argument("--fold", type=int, default=None, metavar="N",
                        help="Run only this fold (1-10). Omit for all folds.")

    # Mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--official", dest="official_mode", action="store_true", default=True,
                            help="Official mode: frozen encoder + trainable query token (default)")
    mode_group.add_argument("--fast", dest="official_mode", action="store_false",
                            help="Fast mode: pre-computed embeddings")

    # Pooling
    parser.add_argument("--pooling", choices=["no", "last", "last_avg"], default="no",
                        help="Pooling mode for official mode (default: no)")

    # Window
    parser.add_argument("--window", type=float, default=10.0, metavar="S",
                        help="Window length in seconds (default: 10)")
    parser.add_argument("--stride", type=float, default=10.0, metavar="S",
                        help="Stride in seconds (default: 10, non-overlapping)")

    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs (default: 50)")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate (default: 5e-3)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")

    # Regularization toggles
    parser.add_argument("--no-mixup", dest="use_mixup", action="store_false", default=True,
                        help="Disable mixup augmentation")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false", default=True,
                        help="Disable mixed precision")

    # Fast-mode options
    parser.add_argument("--normalize", action="store_true", default=False,
                        help="L2 normalize features before classifier (fast mode)")
    parser.add_argument("--no-pooling", dest="use_pooling_fast", action="store_false", default=True,
                        help="Bypass attention pooling in fast mode")
    parser.add_argument("--no-pool-mode", choices=["mean", "flat"], default="mean",
                        help="Flatten mode when --no-pooling (fast mode)")

    # Generalization
    parser.add_argument("--generalization", action="store_true", default=False,
                        help="Stimulus-generalization evaluation")
    parser.add_argument("--gen-seeds", type=int, nargs="+", default=[123],
                        help="Seeds for stimulus splits (default: [123])")

    args = parser.parse_args()

    return LPConfig(
        dataset=args.dataset,
        task_mode=args.task,
        fold=args.fold,
        official_mode=args.official_mode,
        pooling=args.pooling,
        window_size=round(args.window * SAMPLING_RATE),
        stride=round(args.stride * SAMPLING_RATE),
        max_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        use_mixup=args.use_mixup,
        use_amp=args.use_amp,
        normalize_features=args.normalize,
        use_pooling=args.use_pooling_fast,
        no_pool_mode=args.no_pool_mode,
        generalization=args.generalization,
        gen_seeds=args.gen_seeds,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)

    cfg = parse_args()

    L.seed_everything(42, workers=True)

    all_subjects = get_all_subjects(cfg.dataset)
    channel_names = get_channel_names(cfg.dataset)

    print(f"\nDataset: {cfg.dataset}  |  subjects: {len(all_subjects)}")
    print(
        f"Device: {DEVICE}  |  task: {cfg.task_mode}  |  mode: {cfg.mode_tag}  |  "
        f"pooling: {cfg.pooling if cfg.official_mode else cfg.pool_tag}"
    )
    print(
        f"Window: {cfg.window_size / SAMPLING_RATE}s ({cfg.window_size} pts)  "
        f"Stride: {cfg.stride / SAMPLING_RATE}s ({cfg.stride} pts)  "
        f"Mixup: {cfg.use_mixup}  AMP: {cfg.use_amp}"
    )

    # Load REVE model once (used by all folds)
    reve_model, pos_tensor = load_reve_and_positions(channel_names, device=DEVICE)

    # Pre-compute embeddings if fast mode
    if not cfg.official_mode:
        precompute_all_subjects(cfg, all_subjects, reve_model, pos_tensor)

    # Fold splits
    folds = get_kfold_splits(all_subjects)
    folds_to_run = (
        [(cfg.fold, folds[cfg.fold - 1])]
        if cfg.fold is not None
        else [(i + 1, folds[i]) for i in range(N_FOLDS)]
    )

    # Seed list (generalization mode)
    seed_list: list[int | None] = [None]
    if cfg.generalization:
        seed_list = cfg.gen_seeds
        print(f"\nGeneralization mode: {len(seed_list)} seed(s) = {seed_list}")

    # Outer loop: seeds, inner loop: folds
    seed_summaries: list[dict] = []

    for seed in seed_list:
        train_stimuli: set[int] | None = None
        val_stimuli: set[int] | None = None
        gen_seed: int | None = None

        if seed is not None:
            gen_seed = seed
            train_stimuli, val_stimuli = get_stimulus_generalization_split(cfg.task_mode, seed=seed)
            print(f"\n{'=' * COL_W}")
            print(f"  SEED {seed}  |  {len(train_stimuli)} train stimuli, {len(val_stimuli)} held-out")
            print(f"{'=' * COL_W}")

        fold_results: list[dict] = []
        for fold_idx, (train_idx, val_idx) in folds_to_run:
            train_subjects = [all_subjects[i] for i in train_idx]
            val_subjects   = [all_subjects[i] for i in val_idx]

            if cfg.official_mode:
                result = run_fold_official(
                    cfg, fold_idx, train_subjects, val_subjects,
                    reve_model, pos_tensor,
                    train_stimuli=train_stimuli, val_stimuli=val_stimuli,
                    gen_seed=gen_seed,
                )
            else:
                result = run_fold_fast(
                    cfg, fold_idx, train_subjects, val_subjects,
                    train_stimuli=train_stimuli, val_stimuli=val_stimuli,
                    gen_seed=gen_seed,
                )
            fold_results.append(result)

        if len(fold_results) > 1:
            print_fold_summary(cfg, fold_results, gen_seed=gen_seed)

        if gen_seed is not None:
            accs     = [r["val_acc"]     for r in fold_results if r.get("val_acc")     is not None]
            bal_accs = [r["val_bal_acc"] for r in fold_results if r.get("val_bal_acc") is not None]
            aurocs   = [r["val_auroc"]   for r in fold_results if r.get("val_auroc")   is not None]
            f1s      = [r["val_f1"]      for r in fold_results if r.get("val_f1")      is not None]
            seed_summaries.append({
                "seed":         gen_seed,
                "mean_acc":     round(statistics.mean(accs),     4) if accs     else None,
                "mean_bal_acc": round(statistics.mean(bal_accs), 4) if bal_accs else None,
                "mean_auroc":   round(statistics.mean(aurocs),   4) if aurocs   else None,
                "mean_f1":      round(statistics.mean(f1s),      4) if f1s      else None,
                "folds":        fold_results,
            })

    # Cleanup
    del reve_model, pos_tensor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nAll folds complete.")

    if len(seed_summaries) > 1:
        print_cross_seed_summary(cfg, seed_summaries)


if __name__ == "__main__":
    main()
