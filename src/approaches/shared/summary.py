"""
summary.py — Cross-fold and cross-seed summary printing and JSON saving.

Generic module used by both LP and FT pipelines. Accepts any config object
that exposes .dataset, .task_mode, .pool_tag, .window_tag properties.
"""

from __future__ import annotations

import datetime
import json
import math
import statistics
from pathlib import Path

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

COL_W = 90


def fmt_metric(val, width: int = 8, decimals: int = 4) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return f"{'n/a':>{width}}"
    return f"{val:>{width}.{decimals}f}"


def _stat(vals: list[float]) -> tuple[str, str]:
    if not vals:
        return "n/a", "n/a"
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{mean:.4f}", f"{std:.4f}"


def _pooled_classification_report(fold_results: list[dict]) -> dict | None:
    """Pool y_true/y_pred across folds and compute per-class P/R/F1 + confusion matrix.

    Returns None if no fold has predictions saved. In 10-fold cross-subject CV each
    sample appears in exactly one val split, so pooling gives one prediction per
    window across the full dataset.
    """
    y_true: list[int] = []
    y_pred: list[int] = []
    for r in fold_results:
        if r.get("y_true") is not None and r.get("y_pred") is not None:
            y_true.extend(r["y_true"])
            y_pred.extend(r["y_pred"])
    if not y_true:
        return None

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )

    return {
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "per_class": {
            "precision": [round(float(p), 4) for p in precision],
            "recall": [round(float(r), 4) for r in recall],
            "f1": [round(float(f), 4) for f in f1],
            "support": [int(s) for s in support],
        },
        "macro": {
            "precision": round(float(macro_p), 4),
            "recall": round(float(macro_r), 4),
            "f1": round(float(macro_f1), 4),
        },
        "n_samples": len(y_true),
    }


def _print_classification_report(report: dict) -> None:
    print(f"\n{'=' * COL_W}")
    print(f"  POOLED CLASSIFICATION REPORT  (n={report['n_samples']} windows)")
    print(f"{'=' * COL_W}")
    header = f"{'Class':>6}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}  {'Support':>9}"
    print(header)
    print("-" * len(header))
    pc = report["per_class"]
    for i, lbl in enumerate(report["labels"]):
        print(
            f"{lbl:>6}  {pc['precision'][i]:>10.4f}  {pc['recall'][i]:>10.4f}  "
            f"{pc['f1'][i]:>10.4f}  {pc['support'][i]:>9d}"
        )
    print("-" * len(header))
    m = report["macro"]
    print(
        f"{'Macro':>6}  {m['precision']:>10.4f}  {m['recall']:>10.4f}  {m['f1']:>10.4f}"
    )
    print(f"{'=' * COL_W}")


def print_fold_summary(
    cfg,
    fold_results: list[dict],
    *,
    output_dir: Path,
    approach_label: str,
    filename_stem: str,
    extra_json: dict | None = None,
    gen_seed: int | None = None,
) -> None:
    """Print per-fold metrics table and save aggregate JSON.

    Args:
        cfg: Config with .dataset, .task_mode, .pool_tag attributes.
        fold_results: List of per-fold result dicts.
        output_dir: Directory for the summary JSON.
        approach_label: Label for the header (e.g. "official", "LoRA r=16").
        filename_stem: Base filename for the JSON (without .json).
        extra_json: Additional keys to include in the JSON root.
        gen_seed: If set, appended to the header.
    """
    seed_label = f"  |  gen_seed={gen_seed}" if gen_seed is not None else ""

    print(f"\n{'=' * COL_W}")
    print(
        f"  CROSS-FOLD SUMMARY  ({len(fold_results)} folds  |  {cfg.dataset}  |  "
        f"task={cfg.task_mode}  |  {cfg.pool_tag}  |  {approach_label}{seed_label})"
    )
    print(f"{'=' * COL_W}")

    col = (
        f"{'Fold':>5}  {'TrLoss':>8}  {'ValAcc':>8}  {'ValBalAcc':>10}  {'ValAUROC':>9}  "
        f"{'ValF1w':>8}  {'Epochs':>7}  {'BestEp':>7}"
    )
    print(col)
    print("-" * len(col))

    train_losses, accs, bal_accs, aurocs, f1s = [], [], [], [], []
    for r in fold_results:
        tl = r.get("train_loss")
        acc = r.get("val_acc")
        bal_acc = r.get("val_bal_acc")
        auroc = r.get("val_auroc")
        f1 = r.get("val_f1")
        if tl is not None:
            train_losses.append(tl)
        if acc is not None:
            accs.append(acc)
        if bal_acc is not None:
            bal_accs.append(bal_acc)
        if auroc is not None:
            aurocs.append(auroc)
        if f1 is not None:
            f1s.append(f1)

        print(
            f"{r['fold']:>5}  "
            f"{fmt_metric(tl):>8}  "
            f"{fmt_metric(acc):>8}  "
            f"{fmt_metric(bal_acc):>10}  "
            f"{fmt_metric(auroc):>9}  "
            f"{fmt_metric(f1):>8}  "
            f"{r.get('epochs_trained', 0):>7}  "
            f"{r.get('best_epoch', 0):>7}"
        )

    print("-" * len(col))

    tl_mean, tl_std = _stat(train_losses)
    acc_mean, acc_std = _stat(accs)
    bal_mean, bal_std = _stat(bal_accs)
    aur_mean, aur_std = _stat(aurocs)
    f1_mean, f1_std = _stat(f1s)

    print(f"{'Mean':>5}  {tl_mean:>8}  {acc_mean:>8}  {bal_mean:>10}  {aur_mean:>9}  {f1_mean:>8}")
    print(f"{'Std':>5}  {tl_std:>8}  {acc_std:>8}  {bal_std:>10}  {aur_std:>9}  {f1_std:>8}")
    print(f"{'=' * COL_W}")

    # Pooled per-class report (only when folds carry y_true/y_pred)
    classification_report = _pooled_classification_report(fold_results)
    predictions_by_fold: list[dict] | None = None
    if classification_report is not None:
        _print_classification_report(classification_report)
        # Move raw predictions out of per-fold rows into a dedicated top-level
        # field so per-fold rows stay compact but predictions remain available
        # for post-hoc analysis (e.g. subject-level breakdowns).
        predictions_by_fold = [
            {"fold": r["fold"], "y_true": r["y_true"], "y_pred": r["y_pred"]}
            for r in fold_results
            if r.get("y_true") is not None and r.get("y_pred") is not None
        ]
        fold_results = [
            {k: v for k, v in r.items() if k not in ("y_true", "y_pred")}
            for r in fold_results
        ]

    # Save aggregate JSON. When fold results carry predictions, append a
    # `_confmat` suffix so the new (richer) summary doesn't clobber any
    # existing summary file produced by an older run at the same config.
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_confmat" if classification_report is not None else ""
    agg_path = output_dir / f"{filename_stem}{suffix}.json"
    agg = {
        "dataset": cfg.dataset,
        "task_mode": cfg.task_mode,
        "completed_at": datetime.datetime.now().isoformat(),
        "n_folds_run": len(fold_results),
        "mean": {
            "train_loss": round(statistics.mean(train_losses), 4) if train_losses else None,
            "val_acc": round(statistics.mean(accs), 4) if accs else None,
            "val_bal_acc": round(statistics.mean(bal_accs), 4) if bal_accs else None,
            "val_auroc": round(statistics.mean(aurocs), 4) if aurocs else None,
            "val_f1": round(statistics.mean(f1s), 4) if f1s else None,
        },
        "std": {
            "train_loss": round(statistics.stdev(train_losses), 4)
            if len(train_losses) > 1
            else 0.0,
            "val_acc": round(statistics.stdev(accs), 4) if len(accs) > 1 else 0.0,
            "val_bal_acc": round(statistics.stdev(bal_accs), 4) if len(bal_accs) > 1 else 0.0,
            "val_auroc": round(statistics.stdev(aurocs), 4) if len(aurocs) > 1 else 0.0,
            "val_f1": round(statistics.stdev(f1s), 4) if len(f1s) > 1 else 0.0,
        },
        "folds": fold_results,
    }
    if classification_report is not None:
        agg["classification_report"] = classification_report
        agg["predictions_by_fold"] = predictions_by_fold
    if extra_json:
        agg.update(extra_json)
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    print(f"Aggregate results saved -> {agg_path}")


def print_cross_seed_summary(
    cfg,
    seed_summaries: list[dict],
    *,
    output_dir: Path,
    approach_label: str,
    filename_stem: str,
    extra_json: dict | None = None,
) -> None:
    """Print aggregate statistics across multiple generalization seeds."""
    print(f"\n{'=' * COL_W}")
    print(
        f"  CROSS-SEED SUMMARY  ({len(seed_summaries)} seeds  |  {cfg.dataset}  |  "
        f"task={cfg.task_mode}  |  {cfg.pool_tag}  |  {approach_label})"
    )
    print(f"{'=' * COL_W}")

    col = f"{'Seed':>6}  {'MeanAcc':>9}  {'MeanBalAcc':>11}  {'MeanF1':>9}"
    print(col)
    print("-" * len(col))

    accs, bal_accs, f1s = [], [], []
    for s in seed_summaries:
        acc = s.get("mean_acc")
        bal_acc = s.get("mean_bal_acc")
        f1 = s.get("mean_f1")
        if acc is not None:
            accs.append(acc)
        if bal_acc is not None:
            bal_accs.append(bal_acc)
        if f1 is not None:
            f1s.append(f1)
        print(
            f"{s['seed']:>6}  {fmt_metric(acc):>9}  {fmt_metric(bal_acc):>11}  {fmt_metric(f1):>9}"
        )

    print("-" * len(col))

    acc_mean, acc_std = _stat(accs)
    bal_mean, bal_std = _stat(bal_accs)
    f1_mean, f1_std = _stat(f1s)

    print(f"{'Mean':>6}  {acc_mean:>9}  {bal_mean:>11}  {f1_mean:>9}")
    print(f"{'Std':>6}  {acc_std:>9}  {bal_std:>11}  {f1_std:>9}")
    print(f"{'=' * COL_W}")

    output_dir.mkdir(parents=True, exist_ok=True)
    agg_path = output_dir / f"{filename_stem}.json"
    agg = {
        "dataset": cfg.dataset,
        "task_mode": cfg.task_mode,
        "completed_at": datetime.datetime.now().isoformat(),
        "n_seeds": len(seed_summaries),
        "seeds": [s["seed"] for s in seed_summaries],
        "mean": {
            "val_acc": round(statistics.mean(accs), 4) if accs else None,
            "val_bal_acc": round(statistics.mean(bal_accs), 4) if bal_accs else None,
            "val_f1": round(statistics.mean(f1s), 4) if f1s else None,
        },
        "std": {
            "val_acc": round(statistics.stdev(accs), 4) if len(accs) > 1 else 0.0,
            "val_bal_acc": round(statistics.stdev(bal_accs), 4) if len(bal_accs) > 1 else 0.0,
            "val_f1": round(statistics.stdev(f1s), 4) if len(f1s) > 1 else 0.0,
        },
        "per_seed": seed_summaries,
    }
    if extra_json:
        agg.update(extra_json)
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    print(f"Cross-seed aggregate saved -> {agg_path}")
