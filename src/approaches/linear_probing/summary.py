"""
summary.py — Cross-fold and cross-seed summary printing and JSON saving for LP.

Provides:
  - print_fold_summary:       per-fold metrics table + aggregate JSON.
  - print_cross_seed_summary: per-seed aggregate table + cross-seed JSON.
"""

from __future__ import annotations

import datetime
import json
import math
import statistics
from pathlib import Path

from src.approaches.linear_probing.config import LPConfig, OUTPUT_DIR


COL_W = 90


def fmt_metric(val, width: int = 8, decimals: int = 4) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return f"{'n/a':>{width}}"
    return f"{val:>{width}.{decimals}f}"


def print_fold_summary(
    cfg: LPConfig,
    fold_results: list[dict],
    gen_seed: int | None = None,
) -> None:
    """Print a per-fold metrics table and save aggregate JSON."""
    seed_label = f"  |  gen_seed={gen_seed}" if gen_seed is not None else ""

    print(f"\n{'=' * COL_W}")
    print(
        f"  CROSS-FOLD SUMMARY  ({len(fold_results)} folds  |  {cfg.dataset}  |  "
        f"task={cfg.task_mode}  |  {cfg.pool_tag}  |  {cfg.mode_tag}{seed_label})"
    )
    print(f"{'=' * COL_W}")

    col = (
        f"{'Fold':>5}  {'ValAcc':>8}  {'ValBalAcc':>10}  {'ValAUROC':>9}  "
        f"{'ValF1w':>8}  {'Epochs':>7}  {'BestEp':>7}"
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

        print(
            f"{r['fold']:>5}  "
            f"{fmt_metric(acc):>8}  "
            f"{fmt_metric(bal_acc):>10}  "
            f"{fmt_metric(auroc):>9}  "
            f"{fmt_metric(f1):>8}  "
            f"{r.get('epochs_trained', 0):>7}  "
            f"{r.get('best_epoch', 0):>7}"
        )

    print("-" * len(col))

    acc_mean,   acc_std   = _stat(accs)
    bal_mean,   bal_std   = _stat(bal_accs)
    aur_mean,   aur_std   = _stat(aurocs)
    f1_mean,    f1_std    = _stat(f1s)

    print(f"{'Mean':>5}  {acc_mean:>8}  {bal_mean:>10}  {aur_mean:>9}  {f1_mean:>8}")
    print(f"{'Std':>5}  {acc_std:>8}  {bal_std:>10}  {aur_std:>9}  {f1_std:>8}")
    print(f"{'=' * COL_W}")

    # Save aggregate JSON
    gen_tag_name = f"_gen_s{gen_seed}" if gen_seed is not None else ""
    agg_path = (
        OUTPUT_DIR / f"summary_{cfg.dataset}_{cfg.task_mode}_{cfg.window_tag}_"
        f"{cfg.pool_tag}_{cfg.mode_tag}{cfg.mixup_tag}{gen_tag_name}.json"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    agg = {
        "dataset":      cfg.dataset,
        "task_mode":    cfg.task_mode,
        "mode":         cfg.mode_tag,
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
    cfg: LPConfig,
    seed_summaries: list[dict],
) -> None:
    """Print aggregate statistics across multiple generalization seeds."""
    print(f"\n{'=' * COL_W}")
    print(
        f"  CROSS-SEED SUMMARY  ({len(seed_summaries)} seeds  |  {cfg.dataset}  |  "
        f"task={cfg.task_mode}  |  {cfg.pool_tag}  |  {cfg.mode_tag})"
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

    acc_mean,   acc_std   = _stat(accs)
    bal_mean,   bal_std   = _stat(bal_accs)
    f1_mean,    f1_std    = _stat(f1s)

    print(f"{'Mean':>6}  {acc_mean:>9}  {bal_mean:>11}  {f1_mean:>9}")
    print(f"{'Std':>6}  {acc_std:>9}  {bal_std:>11}  {f1_std:>9}")
    print(f"{'=' * COL_W}")

    agg_path = (
        OUTPUT_DIR / f"summary_{cfg.dataset}_{cfg.task_mode}_{cfg.window_tag}_"
        f"{cfg.pool_tag}_{cfg.mode_tag}{cfg.mixup_tag}_gen_cross_seed.json"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    agg = {
        "dataset":      cfg.dataset,
        "task_mode":    cfg.task_mode,
        "mode":         cfg.mode_tag,
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


def _stat(vals: list[float]) -> tuple[str, str]:
    if not vals:
        return "n/a", "n/a"
    mean = statistics.mean(vals)
    std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{mean:.4f}", f"{std:.4f}"
