"""
train_ft.py — Main execution script for Fine-Tuning (LoRA or full FT).

Two-stage pipeline:
  Stage 1 (LP warmup): frozen encoder, train cls_query_token + linear head.
  Stage 2 (FT):        LoRA adapters on encoder (default), or full encoder unfreeze (--fullft).

Evaluation modes (mutually exclusive):
  Default:          10-fold cross-subject CV → saves summary_*.json
  --revesplit:      Official REVE static split (FACED only) → train 0-79 / val 80-99 / test 100-122
                    Single run, saves summary_*_revesplit.json with val + test metrics.
  --generalization: 2/3 stimuli per emotion for train, 1/3 held-out (applied across all folds).

Run with:
    # All folds, FACED, 9-class (LoRA, default)
    uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --task 9-class

    # Single fold
    uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --fold 1

    # Custom LoRA rank
    uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --lora-rank 8

    # Full fine-tuning (no LoRA — unfreezes entire encoder in stage 2)
    uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --fullft

    # Official REVE static split with held-out test evaluation
    uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --revesplit
    uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --fullft --revesplit

    # Generalization evaluation
    uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --generalization

    # Smoke test (few epochs)
    uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --fold 1 --lp-epochs 2 --ft-epochs 3
"""

from __future__ import annotations

import argparse
import copy
import datetime
import gc
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
import wandb

from src.datasets.folds import (
    get_all_subjects, get_kfold_splits, get_stimulus_generalization_split,
    get_official_split, N_FOLDS,
)

from src.approaches.shared.metrics import evaluate_model
from src.approaches.shared.dataset import build_raw_dataset
from src.approaches.shared.training_utils import fmt_dur, COL_W
from src.approaches.shared.reve import load_reve_and_positions, get_channel_names

from src.approaches.fine_tuning.config import (
    FTConfig, OUTPUT_DIR,
    USE_WANDB, WANDB_PROJECT, WANDB_ENTITY,
    DEVICE, NUM_WORKERS, SAMPLING_RATE,
)
from src.approaches.fine_tuning.model import ReveClassifierFT
from src.approaches.fine_tuning.lora import apply_lora, print_lora_summary
from src.approaches.fine_tuning.training import train_stage
from src.approaches.fine_tuning.summary import print_fold_summary, print_cross_seed_summary


# Per-fold runner

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
    test_subject_ids: list[int] | None = None,
) -> dict:
    """
    Run one fold: LP warmup → LoRA (or full) fine-tuning.
    """

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

    # Deep-copy encoder weights so each fold starts from the pretrained state.
    # apply_lora() mutates the model in-place via get_peft_model(), and full FT
    # updates encoder weights directly — both would corrupt subsequent folds if
    # we shared the same reve_model reference.
    reve_for_fold = copy.deepcopy(reve_model)
    pos_for_fold  = pos_tensor.clone()

    # Build datasets
    t_load = time.time()
    print("Building datasets ...", end="  ", flush=True)
    train_ds = build_raw_dataset(cfg, train_subject_ids, stimulus_filter=train_stimuli)
    val_ds   = build_raw_dataset(cfg, val_subject_ids, stimulus_filter=val_stimuli)
    print(
        f"done in {fmt_dur(time.time() - t_load)}  "
        f"(train={len(train_ds):,}  val={len(val_ds):,})"
    )

    test_ds = None
    if test_subject_ids is not None:
        test_ds = build_raw_dataset(cfg, test_subject_ids)
        print(f"  test={len(test_ds):,} windows")

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
        reve_model=reve_for_fold,
        pos_tensor=pos_for_fold,
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

    # Stage 1: LP warmup
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
        save_trainable_only=True,
    )
    lp_epochs = lp_result["epochs_trained"]

    # Restore best LP state before moving to FT (partial state — trainable only)
    if lp_result.get("best_state"):
        model.load_state_dict(lp_result["best_state"], strict=False)
        model.to(DEVICE)

    # Stage 2: LoRA or full fine-tuning
    model.unfreeze_encoder()
    if cfg.full_ft:
        model.set_dropout(cfg.ft_dropout)
        print(f"\n>>> Stage 2: Full fine-tuning  |  trainable params: {model.n_trainable_params():,}")
    else:
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

        if cfg.full_ft:
            model_path = ckpt_dir / "full_model.pt"
            torch.save(ft_result["best_state"], model_path)
            print(f"Full model saved → {model_path}")
        else:
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

    # Test evaluation (revesplit only)
    test_metrics = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        print(f"\n>>> Test evaluation ({len(test_ds):,} windows)")
        test_metrics = evaluate_model(
            model, test_loader, device=DEVICE,
            n_classes=cfg.num_classes, use_amp=cfg.use_amp,
        )
        print(
            f"  test_acc={test_metrics['accuracy']:.4f}  "
            f"test_bal_acc={test_metrics['balanced_acc']:.4f}  "
            f"test_auroc={test_metrics['auroc']:.4f}  "
            f"test_f1={test_metrics['f1_weighted']:.4f}"
        )

    if USE_WANDB:
        best_log = {
            "best/train_loss":  ft_result.get("train_loss"),
            "best/val_loss":    ft_result.get("val_loss"),
            "best/val_acc":     ft_result.get("val_acc"),
            "best/val_bal_acc": ft_result.get("val_bal_acc"),
            "best/val_auroc":   ft_result.get("val_auroc"),
            "best/val_f1":      ft_result.get("val_f1"),
        }
        if test_metrics is not None:
            best_log.update({
                "test/acc":     test_metrics["accuracy"],
                "test/bal_acc": test_metrics["balanced_acc"],
                "test/auroc":   test_metrics["auroc"],
                "test/f1":      test_metrics["f1_weighted"],
            })
        wandb.log(best_log)
        wandb.finish()

    # Cleanup
    to_del = [model, reve_for_fold, pos_for_fold, train_ds, val_ds, train_loader, val_loader]
    if test_ds is not None:
        to_del.append(test_ds)
    del to_del
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "fold":              fold_idx,
        "train_loss":        ft_result.get("train_loss"),
        "val_loss":          ft_result.get("val_loss"),
        "val_acc":           ft_result.get("val_acc"),
        "val_bal_acc":       ft_result.get("val_bal_acc"),
        "val_auroc":         ft_result.get("val_auroc"),
        "val_f1":            ft_result.get("val_f1"),
        "best_epoch":        ft_result.get("best_epoch"),
        "lp_epochs_trained": lp_epochs,
        "ft_epochs_trained": ft_result.get("epochs_trained"),
        "lp_train_loss":     lp_result.get("train_loss"),
        "lp_val_acc":        lp_result.get("val_acc"),
    }
    if test_metrics is not None:
        result.update({
            "test_acc":     test_metrics["accuracy"],
            "test_bal_acc": test_metrics["balanced_acc"],
            "test_auroc":   test_metrics["auroc"],
            "test_f1":      test_metrics["f1_weighted"],
        })
    return result


# CLI

def parse_args() -> FTConfig:
    _d = FTConfig()  # single source of truth for all defaults
    _w = round(_d.window_size / SAMPLING_RATE)
    _st = round(_d.stride / SAMPLING_RATE)

    parser = argparse.ArgumentParser(description="Fine-Tuning with LoRA on REVE")

    parser.add_argument("--dataset", choices=["faced", "thu-ep"], default=_d.dataset)
    parser.add_argument("--task", choices=["binary", "9-class"], default=_d.task_mode,
                        help="Classification task")
    parser.add_argument("--fold", type=int, default=None, metavar="N",
                        help="Run only this fold (1-10). Omit for all folds.")

    # FT mode
    parser.add_argument("--fullft", action="store_true", default=_d.full_ft,
                        help="Full fine-tuning (no LoRA)")
    parser.add_argument("--revesplit", action="store_true", default=_d.reve_split,
                        help="Official REVE split: train 0-79, val 80-99, test 100-122 (FACED only)")

    # Pooling
    parser.add_argument("--pooling", choices=["no", "last", "last_avg"], default=_d.pooling)

    # Window
    parser.add_argument("--window", type=float, default=_w, metavar="S",
                        help=f"Window length in seconds (default: {_w})")
    parser.add_argument("--stride", type=float, default=_st, metavar="S",
                        help=f"Stride in seconds (default: {_st})")

    # Shared training
    parser.add_argument("--batch-size", type=int, default=_d.batch_size)
    parser.add_argument("--no-mixup", dest="use_mixup", action="store_false",
                        default=_d.use_mixup, help="Disable mixup augmentation")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false",
                        default=_d.use_amp, help="Disable mixed precision")

    # LP stage
    parser.add_argument("--lp-epochs", type=int, default=_d.lp_max_epochs,
                        help=f"LP max epochs (default: {_d.lp_max_epochs})")
    parser.add_argument("--lp-lr", type=float, default=_d.lp_lr,
                        help=f"LP learning rate (default: {_d.lp_lr})")

    # FT stage
    parser.add_argument("--ft-epochs", type=int, default=_d.ft_max_epochs,
                        help=f"FT max epochs (default: {_d.ft_max_epochs})")
    parser.add_argument("--ft-lr", type=float, default=_d.ft_lr,
                        help=f"FT learning rate (default: {_d.ft_lr})")

    # LoRA
    parser.add_argument("--lora-rank", type=int, default=_d.lora_rank,
                        help=f"LoRA rank (default: {_d.lora_rank})")
    parser.add_argument("--lora-alpha", type=int, default=None,
                        help="LoRA alpha (default: same as rank)")
    parser.add_argument("--lora-target", choices=["attention", "attention+ffn"],
                        default=_d.lora_target)

    # Generalization
    parser.add_argument("--generalization", action="store_true", default=_d.generalization,
                        help="Stimulus-generalization evaluation")
    parser.add_argument("--gen-seeds", type=int, nargs="+", default=_d.gen_seeds,
                        help="Seeds for stimulus splits")

    args = parser.parse_args()

    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_rank

    if args.revesplit and args.dataset != "faced":
        parser.error("--revesplit is only supported for FACED dataset")
    if args.revesplit and args.generalization:
        parser.error("--revesplit and --generalization are mutually exclusive")
    if args.revesplit and args.fold is not None:
        parser.error("--revesplit and --fold are mutually exclusive")

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
        full_ft=args.fullft,
        reve_split=args.revesplit,
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


# Main

def main() -> None:
    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)

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
    if cfg.full_ft:
        print("  ** fullft mode: full fine-tuning (no LoRA) **")
    if cfg.reve_split:
        print("  ** revesplit mode: official REVE train/val/test split **")
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

    # REVE split path (single run with test set, match official REVE evaluation protocol on FACED)
    if cfg.reve_split:
        train_subjects, val_subjects, test_subjects = get_official_split(cfg.dataset)
        print(
            f"\nREVE split: train={len(train_subjects)}  "
            f"val={len(val_subjects)}  test={len(test_subjects)}"
        )

        result = run_fold_ft(
            cfg, 1, train_subjects, val_subjects,
            reve_model, pos_tensor,
            test_subject_ids=test_subjects,
        )

        # Save summary JSON for revesplit
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        summary = {
            "dataset": cfg.dataset,
            "task_mode": cfg.task_mode,
            "approach": "ft_fullft" if cfg.full_ft else "ft_lora",
            "lora_rank": cfg.lora_rank,
            "split": "revesplit",
            "completed_at": datetime.datetime.now().isoformat(),
            "val": {
                "acc": result.get("val_acc"),
                "bal_acc": result.get("val_bal_acc"),
                "auroc": result.get("val_auroc"),
                "f1": result.get("val_f1"),
                "loss": result.get("val_loss"),
            },
            "test": {
                "acc": result.get("test_acc"),
                "bal_acc": result.get("test_bal_acc"),
                "auroc": result.get("test_auroc"),
                "f1": result.get("test_f1"),
            },
            "train_loss": result.get("train_loss"),
            "best_epoch": result.get("best_epoch"),
            "lp_epochs_trained": result.get("lp_epochs_trained"),
            "ft_epochs_trained": result.get("ft_epochs_trained"),
        }
        ft_tag = "fullft" if cfg.full_ft else f"r{cfg.lora_rank}"
        summary_path = OUTPUT_DIR / (
            f"summary_{cfg.dataset}_{cfg.task_mode}_{cfg.window_tag}_"
            f"{cfg.pool_tag}_{ft_tag}{cfg.mixup_tag}_revesplit.json"
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved -> {summary_path}")

        del reve_model, pos_tensor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nDone.")
        return

    # Standard k-fold path
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
