"""
train_ft.py — Main execution script for Fine-Tuning with LoRA.

Two-stage pipeline:
  Stage 1 (LP warmup): frozen encoder, train cls_query_token + linear head.
  Stage 2 (FT):        LoRA adapters on encoder, train LoRA + head + query token.

Run with:
    # All folds, FACED, 9-class
    uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --task 9-class

    # Single fold
    uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --fold 1

    # Custom LoRA rank
    uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --lora-rank 8

    # Smoke test (few epochs)
    uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --fold 1 --lp-epochs 2 --ft-epochs 3
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
from torch.utils.data import DataLoader
import wandb

from src.datasets.folds import (
    get_all_subjects, get_kfold_splits, get_stimulus_generalization_split, N_FOLDS,
)

from src.approaches.linear_probing.model import (
    load_reve_and_positions, evaluate_model,
)
from src.approaches.linear_probing.stable_adamw import StableAdamW
from src.approaches.linear_probing.train_lp import (
    build_raw_dataset, get_channel_names,
    PatienceMonitor, _get_exponential_warmup_lambda,
    fmt_dur, fmt_metric, COL_W,
)

from src.approaches.fine_tuning.config import (
    FTConfig, OUTPUT_DIR,
    USE_WANDB, WANDB_PROJECT, WANDB_ENTITY,
    DEVICE, NUM_WORKERS, SAMPLING_RATE,
)
from src.approaches.fine_tuning.model import ReveClassifierFT
from src.approaches.fine_tuning.lora import apply_lora, print_lora_summary


# ── Summary helpers (delegate to LP summary with FT output dir) ──────────────

def print_fold_summary(
    cfg: FTConfig,
    fold_results: list[dict],
    gen_seed: int | None = None,
) -> None:
    """Print per-fold metrics table and save aggregate JSON for FT runs."""
    import datetime
    import json
    import math

    seed_label = f"  |  gen_seed={gen_seed}" if gen_seed is not None else ""

    print(f"\n{'=' * COL_W}")
    print(
        f"  CROSS-FOLD SUMMARY  ({len(fold_results)} folds  |  {cfg.dataset}  |  "
        f"task={cfg.task_mode}  |  {cfg.pool_tag}  |  LoRA r={cfg.lora_rank}{seed_label})"
    )
    print(f"{'=' * COL_W}")

    col = (
        f"{'Fold':>5}  {'ValAcc':>8}  {'ValBalAcc':>10}  {'ValAUROC':>9}  "
        f"{'ValF1w':>8}  {'LPEp':>6}  {'FTEp':>6}  {'BestEp':>7}"
    )
    print(col)
    print("-" * len(col))

    accs, bal_accs, aurocs, f1s = [], [], [], []
    for r in fold_results:
        acc     = r.get("val_acc")
        bal_acc = r.get("val_bal_acc")
        auroc   = r.get("val_auroc")
        f1      = r.get("val_f1")
        if acc     is not None: accs.append(acc)
        if bal_acc is not None: bal_accs.append(bal_acc)
        if auroc   is not None: aurocs.append(auroc)
        if f1      is not None: f1s.append(f1)

        def _fmt(val, w=8):
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return f"{'n/a':>{w}}"
            return f"{val:>{w}.4f}"

        print(
            f"{r['fold']:>5}  "
            f"{_fmt(acc):>8}  {_fmt(bal_acc):>10}  {_fmt(auroc):>9}  {_fmt(f1):>8}  "
            f"{r.get('lp_epochs_trained', 0):>6}  "
            f"{r.get('ft_epochs_trained', 0):>6}  "
            f"{r.get('best_epoch', 0):>7}"
        )

    print("-" * len(col))

    def _stat(vals):
        if not vals:
            return "n/a", "n/a"
        mean = statistics.mean(vals)
        std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f"{mean:.4f}", f"{std:.4f}"

    acc_mean, acc_std = _stat(accs)
    bal_mean, bal_std = _stat(bal_accs)
    aur_mean, aur_std = _stat(aurocs)
    f1_mean,  f1_std  = _stat(f1s)

    print(f"{'Mean':>5}  {acc_mean:>8}  {bal_mean:>10}  {aur_mean:>9}  {f1_mean:>8}")
    print(f"{'Std':>5}  {acc_std:>8}  {bal_std:>10}  {aur_std:>9}  {f1_std:>8}")
    print(f"{'=' * COL_W}")

    # Save aggregate JSON
    gen_tag_name = f"_gen_s{gen_seed}" if gen_seed is not None else ""
    agg_path = (
        OUTPUT_DIR / f"summary_{cfg.dataset}_{cfg.task_mode}_{cfg.window_tag}_"
        f"{cfg.pool_tag}_r{cfg.lora_rank}{gen_tag_name}.json"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    agg = {
        "dataset":      cfg.dataset,
        "task_mode":    cfg.task_mode,
        "approach":     "ft_lora",
        "lora_rank":    cfg.lora_rank,
        "completed_at": datetime.datetime.now().isoformat(),
        "n_folds_run":  len(fold_results),
        "mean": {
            "val_acc":     round(statistics.mean(accs), 4)     if accs     else None,
            "val_bal_acc": round(statistics.mean(bal_accs), 4) if bal_accs else None,
            "val_auroc":   round(statistics.mean(aurocs), 4)   if aurocs   else None,
            "val_f1":      round(statistics.mean(f1s), 4)      if f1s      else None,
        },
        "std": {
            "val_acc":     round(statistics.stdev(accs), 4)     if len(accs)     > 1 else 0.0,
            "val_bal_acc": round(statistics.stdev(bal_accs), 4) if len(bal_accs) > 1 else 0.0,
            "val_auroc":   round(statistics.stdev(aurocs), 4)   if len(aurocs)   > 1 else 0.0,
            "val_f1":      round(statistics.stdev(f1s), 4)      if len(f1s)      > 1 else 0.0,
        },
        "folds": fold_results,
    }
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    print(f"Aggregate results saved -> {agg_path}")


def print_cross_seed_summary(
    cfg: FTConfig,
    seed_summaries: list[dict],
) -> None:
    """Print aggregate statistics across multiple generalization seeds."""
    print(f"\n{'=' * COL_W}")
    print(
        f"  CROSS-SEED SUMMARY  ({len(seed_summaries)} seeds  |  {cfg.dataset}  |  "
        f"task={cfg.task_mode}  |  {cfg.pool_tag}  |  LoRA r={cfg.lora_rank})"
    )
    print(f"{'=' * COL_W}")

    col = f"{'Seed':>6}  {'MeanAcc':>9}  {'MeanBalAcc':>11}  {'MeanF1':>9}"
    print(col)
    print("-" * len(col))

    accs, bal_accs, f1s = [], [], []
    for s in seed_summaries:
        acc     = s.get("mean_acc")
        bal_acc = s.get("mean_bal_acc")
        f1      = s.get("mean_f1")
        if acc     is not None: accs.append(acc)
        if bal_acc is not None: bal_accs.append(bal_acc)
        if f1      is not None: f1s.append(f1)
        print(
            f"{s['seed']:>6}  "
            f"{fmt_metric(acc):>9}  "
            f"{fmt_metric(bal_acc):>11}  "
            f"{fmt_metric(f1):>9}"
        )

    print("-" * len(col))

    def _stat(vals):
        if not vals:
            return "n/a", "n/a"
        mean = statistics.mean(vals)
        std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f"{mean:.4f}", f"{std:.4f}"

    acc_mean, acc_std = _stat(accs)
    bal_mean, bal_std = _stat(bal_accs)
    f1_mean,  f1_std  = _stat(f1s)

    print(f"{'Mean':>6}  {acc_mean:>9}  {bal_mean:>11}  {f1_mean:>9}")
    print(f"{'Std':>6}  {acc_std:>9}  {bal_std:>11}  {f1_std:>9}")
    print(f"{'=' * COL_W}")

    import datetime
    import json

    agg_path = (
        OUTPUT_DIR / f"summary_{cfg.dataset}_{cfg.task_mode}_{cfg.window_tag}_"
        f"{cfg.pool_tag}_r{cfg.lora_rank}_gen_cross_seed.json"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    agg = {
        "dataset":      cfg.dataset,
        "task_mode":    cfg.task_mode,
        "approach":     "ft_lora",
        "lora_rank":    cfg.lora_rank,
        "completed_at": datetime.datetime.now().isoformat(),
        "n_seeds":      len(seed_summaries),
        "seeds":        [s["seed"] for s in seed_summaries],
        "mean": {
            "val_acc":     round(statistics.mean(accs), 4)     if accs     else None,
            "val_bal_acc": round(statistics.mean(bal_accs), 4) if bal_accs else None,
            "val_f1":      round(statistics.mean(f1s), 4)      if f1s      else None,
        },
        "std": {
            "val_acc":     round(statistics.stdev(accs), 4)     if len(accs)     > 1 else 0.0,
            "val_bal_acc": round(statistics.stdev(bal_accs), 4) if len(bal_accs) > 1 else 0.0,
            "val_f1":      round(statistics.stdev(f1s), 4)      if len(f1s)      > 1 else 0.0,
        },
        "per_seed": seed_summaries,
    }
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    print(f"Cross-seed aggregate saved -> {agg_path}")


# ── Generic training stage ───────────────────────────────────────────────────

def train_stage(
    model: ReveClassifierFT,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    stage_name: str,
    lr: float,
    max_epochs: int,
    warmup_epochs: int,
    scheduler_patience: int,
    early_stop_patience: int,
    grad_clip: float,
    weight_decay: float,
    use_mixup: bool,
    use_amp: bool,
    n_classes: int,
    device: str,
    wandb_epoch_offset: int = 0,
) -> dict:
    """
    Shared training loop for both LP warmup and FT stages.

    Returns dict with best metrics, best state dict, and training stats.
    """
    device_type = "cuda" if "cuda" in device else "cpu"

    optimizer = StableAdamW(
        model.trainable_parameters(),
        lr=lr,
        betas=(0.92, 0.999),
        weight_decay=weight_decay,
    )

    warmup_steps = warmup_epochs * len(train_loader)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_get_exponential_warmup_lambda(warmup_steps),
    )

    reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=scheduler_patience,
    )

    scaler = torch.amp.GradScaler()
    patience_monitor = PatienceMonitor(early_stop_patience)

    best_metrics = {}
    best_acc = 0.0
    best_state = None

    print(f"\n{'─' * COL_W}")
    print(f"  Stage: {stage_name}  |  lr={lr}  |  max_epochs={max_epochs}  |  grad_clip={grad_clip}")
    header = (
        f"{'Epoch':>6}  {'EpTime':>7}  {'Elapsed':>8}  "
        f"{'TrLoss':>8}  {'TrAcc':>7}  {'VaAcc':>7}  {'VaBalAcc':>9}  "
        f"{'VaAUROC':>8}  {'VaF1w':>7}  {'LR':>10}"
    )
    print(header)
    print(f"{'─' * COL_W}")

    fit_start = time.time()
    last_epoch = 0

    for epoch in range(max_epochs):
        last_epoch = epoch
        epoch_start = time.time()
        warmup_active = epoch <= warmup_epochs

        # ── Train ──────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        n_correct = 0
        n_samples = 0

        for batch in train_loader:
            eeg, target = batch[0].to(device), batch[1].long().to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device_type, enabled=use_amp, dtype=torch.float16):
                if use_mixup:
                    mm = random.random()
                    perm = torch.randperm(eeg.size(0), device=device)
                    output = model(mm * eeg + (1 - mm) * eeg[perm])
                    loss = mm * F.cross_entropy(output, target) + (1 - mm) * F.cross_entropy(output, target[perm])
                else:
                    output = model(eeg)
                    loss = F.cross_entropy(output, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                grad_clip,
            )

            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()

            if warmup_active and scale_before == scaler.get_scale():
                warmup_scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            with torch.no_grad():
                preds = output.argmax(dim=1)
            n_correct += (preds == target).sum().item()
            n_samples += target.size(0)

        avg_loss = epoch_loss / max(n_batches, 1)
        train_acc = n_correct / max(n_samples, 1)

        # ── Validate ───────────────────────────────────────────────────
        metrics = evaluate_model(
            model, val_loader, device=device,
            n_classes=n_classes, use_amp=use_amp,
        )

        val_acc = metrics["accuracy"]
        reduce_scheduler.step(val_acc)

        # Track best
        if val_acc > best_acc:
            best_acc = val_acc
            best_metrics = {**metrics, "epoch": epoch + 1}
            # Save all trainable params
            best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

        # Print epoch summary
        ep_time = time.time() - epoch_start
        elapsed = time.time() - fit_start
        current_lr = optimizer.param_groups[0]["lr"]
        avg_ep = elapsed / (epoch + 1)
        remaining = avg_ep * (max_epochs - epoch - 1)
        eta = f"ETA {fmt_dur(remaining)}" if epoch + 1 < max_epochs else "done"

        print(
            f"{epoch+1:>6}  {fmt_dur(ep_time):>7}  {fmt_dur(elapsed):>8}  "
            f"{avg_loss:>8.4f}  {train_acc:>7.4f}  {val_acc:>7.4f}  {metrics['balanced_acc']:>9.4f}  "
            f"{metrics['auroc']:>8.4f}  {metrics['f1_weighted']:>7.4f}  "
            f"{current_lr:>10.2e}  ({eta})"
        )

        # W&B per-epoch logging
        if wandb.run is not None:
            wandb.log({
                f"{stage_name}/train_loss": avg_loss,
                f"{stage_name}/train_acc":  train_acc,
                f"{stage_name}/val_acc":    val_acc,
                f"{stage_name}/val_bal_acc": metrics["balanced_acc"],
                f"{stage_name}/val_auroc":  metrics["auroc"],
                f"{stage_name}/val_f1":     metrics["f1_weighted"],
                f"{stage_name}/lr":         current_lr,
            }, step=wandb_epoch_offset + epoch + 1)

        # Early stopping
        if patience_monitor(val_acc):
            print(f"Early stopping at epoch {epoch + 1} (patience={early_stop_patience})")
            break

    total_time = time.time() - fit_start
    print(f"{'─' * COL_W}")
    print(
        f"{stage_name} complete — {last_epoch + 1} epochs | total: {fmt_dur(total_time)} | "
        f"best epoch={best_metrics.get('epoch', 'n/a')} val_acc={best_acc:.4f}"
    )
    print(f"{'─' * COL_W}")

    return {
        "val_acc":        best_metrics.get("accuracy"),
        "val_bal_acc":    best_metrics.get("balanced_acc"),
        "val_auroc":      best_metrics.get("auroc"),
        "val_f1":         best_metrics.get("f1_weighted"),
        "best_epoch":     best_metrics.get("epoch"),
        "epochs_trained": last_epoch + 1,
        "best_state":     best_state,
    }


# ── Per-fold runner ──────────────────────────────────────────────────────────

def run_fold_ft(
    cfg: FTConfig,
    fold_idx: int,
    train_subject_ids: list[int],
    val_subject_ids: list[int],
    reve_model: torch.nn.Module,
    pos_tensor: torch.Tensor,
    train_stimuli: set[int] | None = None,
    val_stimuli: set[int] | None = None,
    gen_seed: int | None = None,
) -> dict:
    """Run one fold: LP warmup → LoRA fine-tuning."""
    seed_label = f"  |  gen_seed={gen_seed}" if gen_seed is not None else ""
    gen_tag = f"  |  GENERALIZATION{seed_label}" if cfg.generalization else ""

    print(f"\n{'#' * COL_W}")
    print(
        f"  Fold {fold_idx}/{N_FOLDS}  |  {cfg.dataset}  |  task={cfg.task_mode}  |  "
        f"pooling={cfg.pooling}  |  LoRA r={cfg.lora_rank}  |  device={DEVICE}{gen_tag}"
    )
    print(
        f"  train: {len(train_subject_ids)} subjects  |  val: {len(val_subject_ids)} subjects"
    )
    print(f"{'#' * COL_W}")

    # Build datasets
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
    model = ReveClassifierFT(
        reve_model=reve_model,
        pos_tensor=pos_tensor,
        n_classes=cfg.num_classes,
        n_channels=cfg.n_channels,
        window_size=cfg.window_size,
        pooling=cfg.pooling,
        dropout=cfg.lp_dropout,
    )
    model.to(DEVICE)

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

    # ── Stage 1: LP warmup ────────────────────────────────────────────
    model.freeze_encoder()
    model.set_dropout(cfg.lp_dropout)
    print(f"\n>>> Stage 1: LP warmup  |  trainable params: {model.n_trainable_params():,}")

    lp_result = train_stage(
        model, train_loader, val_loader,
        stage_name="lp",
        lr=cfg.lp_lr,
        max_epochs=cfg.lp_max_epochs,
        warmup_epochs=cfg.lp_warmup_epochs,
        scheduler_patience=cfg.lp_scheduler_patience,
        early_stop_patience=cfg.lp_early_stop_patience,
        grad_clip=cfg.lp_grad_clip,
        weight_decay=cfg.weight_decay,
        use_mixup=cfg.use_mixup,
        use_amp=cfg.use_amp,
        n_classes=cfg.num_classes,
        device=DEVICE,
        wandb_epoch_offset=0,
    )
    lp_epochs = lp_result["epochs_trained"]

    # Restore best LP state before moving to FT
    if lp_result.get("best_state"):
        model.load_state_dict(lp_result["best_state"])
        model.to(DEVICE)

    # ── Stage 2: LoRA fine-tuning ─────────────────────────────────────
    model.unfreeze_encoder()
    apply_lora(model, cfg)
    model.set_dropout(cfg.ft_dropout)
    print(f"\n>>> Stage 2: LoRA fine-tuning")
    print_lora_summary(model)

    ft_result = train_stage(
        model, train_loader, val_loader,
        stage_name="ft",
        lr=cfg.ft_lr,
        max_epochs=cfg.ft_max_epochs,
        warmup_epochs=cfg.ft_warmup_epochs,
        scheduler_patience=cfg.ft_scheduler_patience,
        early_stop_patience=cfg.ft_early_stop_patience,
        grad_clip=cfg.ft_grad_clip,
        weight_decay=cfg.weight_decay,
        use_mixup=cfg.use_mixup,
        use_amp=cfg.use_amp,
        n_classes=cfg.num_classes,
        device=DEVICE,
        wandb_epoch_offset=lp_epochs,
    )

    # Save best checkpoint (LoRA adapters + head + cls_query_token)
    ckpt_dir = OUTPUT_DIR / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if ft_result.get("best_state"):
        # Restore best FT state
        model.load_state_dict(ft_result["best_state"])
        model.to(DEVICE)

        # Save LoRA adapter weights (peft convention)
        lora_dir = ckpt_dir / "lora_adapter"
        model.reve.save_pretrained(str(lora_dir))
        print(f"LoRA adapter saved → {lora_dir}")

        # Save head + cls_query_token
        head_state = {
            k: v.cpu().clone() for k, v in model.state_dict().items()
            if k.startswith("cls_query_token") or k.startswith("linear_head")
        }
        head_path = ckpt_dir / "head_weights.pt"
        torch.save(head_state, head_path)
        print(f"Head weights saved → {head_path}")

    if USE_WANDB:
        wandb.log({
            "best/val_acc":     ft_result.get("val_acc"),
            "best/val_bal_acc": ft_result.get("val_bal_acc"),
            "best/val_auroc":   ft_result.get("val_auroc"),
            "best/val_f1":      ft_result.get("val_f1"),
        })
        wandb.finish()

    # Cleanup
    del model, train_ds, val_ds, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "fold":              fold_idx,
        "val_acc":           ft_result.get("val_acc"),
        "val_bal_acc":       ft_result.get("val_bal_acc"),
        "val_auroc":         ft_result.get("val_auroc"),
        "val_f1":            ft_result.get("val_f1"),
        "best_epoch":        ft_result.get("best_epoch"),
        "lp_epochs_trained": lp_epochs,
        "ft_epochs_trained": ft_result.get("epochs_trained"),
        "lp_val_acc":        lp_result.get("val_acc"),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> FTConfig:
    parser = argparse.ArgumentParser(description="Fine-Tuning with LoRA on REVE")

    parser.add_argument("--dataset", choices=["faced", "thu-ep"], default="faced",
                        help="Dataset (default: faced)")
    parser.add_argument("--task", choices=["binary", "9-class"], default="binary",
                        help="Classification task (default: binary)")
    parser.add_argument("--fold", type=int, default=None, metavar="N",
                        help="Run only this fold (1-10). Omit for all folds.")

    # Pooling
    parser.add_argument("--pooling", choices=["no", "last", "last_avg"], default="no",
                        help="Pooling mode (default: no)")

    # Window
    parser.add_argument("--window", type=float, default=10.0, metavar="S",
                        help="Window length in seconds (default: 10)")
    parser.add_argument("--stride", type=float, default=10.0, metavar="S",
                        help="Stride in seconds (default: 10)")

    # Shared training
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--no-mixup", dest="use_mixup", action="store_false", default=True,
                        help="Disable mixup augmentation")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false", default=True,
                        help="Disable mixed precision")

    # LP stage
    parser.add_argument("--lp-epochs", type=int, default=20, help="LP max epochs (default: 20)")
    parser.add_argument("--lp-lr", type=float, default=5e-3, help="LP learning rate (default: 5e-3)")

    # FT stage
    parser.add_argument("--ft-epochs", type=int, default=200, help="FT max epochs (default: 200)")
    parser.add_argument("--ft-lr", type=float, default=1e-4, help="FT learning rate (default: 1e-4)")

    # LoRA
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=None,
                        help="LoRA alpha (default: same as rank)")
    parser.add_argument("--lora-target", choices=["attention", "attention+ffn"], default="attention",
                        help="LoRA target modules (default: attention)")

    # Generalization
    parser.add_argument("--generalization", action="store_true", default=False,
                        help="Stimulus-generalization evaluation")
    parser.add_argument("--gen-seeds", type=int, nargs="+", default=[123],
                        help="Seeds for stimulus splits (default: [123])")

    args = parser.parse_args()

    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_rank

    return FTConfig(
        dataset=args.dataset,
        task_mode=args.task,
        fold=args.fold,
        pooling=args.pooling,
        window_size=round(args.window * SAMPLING_RATE),
        stride=round(args.stride * SAMPLING_RATE),
        batch_size=args.batch_size,
        use_mixup=args.use_mixup,
        use_amp=args.use_amp,
        lp_max_epochs=args.lp_epochs,
        lp_lr=args.lp_lr,
        ft_max_epochs=args.ft_epochs,
        ft_lr=args.ft_lr,
        lora_rank=args.lora_rank,
        lora_alpha=lora_alpha,
        lora_target=args.lora_target,
        generalization=args.generalization,
        gen_seeds=args.gen_seeds,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import lightning as L
    L.seed_everything(42, workers=True)

    cfg = parse_args()

    all_subjects = get_all_subjects(cfg.dataset)
    channel_names = get_channel_names(cfg.dataset)

    print(f"\nDataset: {cfg.dataset}  |  subjects: {len(all_subjects)}")
    print(
        f"Device: {DEVICE}  |  task: {cfg.task_mode}  |  pooling: {cfg.pooling}  |  "
        f"LoRA rank: {cfg.lora_rank}  alpha: {cfg.lora_alpha}"
    )
    print(
        f"Window: {cfg.window_size / SAMPLING_RATE}s ({cfg.window_size} pts)  "
        f"Stride: {cfg.stride / SAMPLING_RATE}s ({cfg.stride} pts)  "
        f"Mixup: {cfg.use_mixup}  AMP: {cfg.use_amp}"
    )
    print(
        f"LP: lr={cfg.lp_lr} epochs={cfg.lp_max_epochs}  |  "
        f"FT: lr={cfg.ft_lr} epochs={cfg.ft_max_epochs}"
    )

    # Load REVE model once
    reve_model, pos_tensor = load_reve_and_positions(channel_names, device=DEVICE)

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

            result = run_fold_ft(
                cfg, fold_idx, train_subjects, val_subjects,
                reve_model, pos_tensor,
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
