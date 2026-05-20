"""
plot_inference.py — Generate plots from a saved inference run.

Reads the JSON + NPZ pair produced by `inference_subject_wise.py` and writes:
  - <stem>_roc.png            : per-class one-vs-rest ROC curves + macro avg
  - <stem>_confmat.png        : normalized confusion matrix heatmap
  - <stem>_per_class.png      : per-class precision / recall / F1 bar chart
  - <stem>_per_subject.png    : per-subject accuracy histogram

Usage:
    uv run python -m src.inference.plot_inference \\
        --dir main-results/lp_9-class \\
        --stem lp_faced_v2_9-class_w10s10_pool_no_official_nomixup

The script looks for <stem>.json and <stem>.npz in --dir (default: cwd).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

# Emotion class names matching the 0..8 labels in the project.
CLASS_NAMES_9 = [
    "Anger", "Disgust", "Fear", "Sadness", "Neutral",
    "Amusement", "Inspiration", "Joy", "Tenderness",
]
CLASS_NAMES_2 = ["Negative", "Positive"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stem", required=True,
                   help="Filename stem (without extension). Files <stem>.json and <stem>.npz must exist.")
    p.add_argument("--dir", type=Path, default=Path.cwd(),
                   help="Directory containing <stem>.json/.npz and where plots are written. "
                        "Default: current working directory.")
    return p.parse_args()


def class_names(n_classes: int) -> list[str]:
    if n_classes == 9:
        return CLASS_NAMES_9
    if n_classes == 2:
        return CLASS_NAMES_2
    return [str(i) for i in range(n_classes)]


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    """Per-class one-vs-rest ROC curves + macro-average curve."""
    n_classes = len(labels)
    names = class_names(n_classes)

    fpr_grid = np.linspace(0.0, 1.0, 200)
    tpr_interp = np.zeros((n_classes, fpr_grid.size))
    aucs = np.zeros(n_classes)

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    for i, lbl in enumerate(labels):
        y_bin = (y_true == lbl).astype(int)
        if y_bin.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        aucs[i] = roc_auc
        tpr_interp[i] = np.interp(fpr_grid, fpr, tpr)
        tpr_interp[i, 0] = 0.0
        ax.plot(fpr, tpr, color=cmap(i), lw=1.5,
                label=f"{names[i]} (AUC={roc_auc:.3f})")

    # Macro-average across classes
    mean_tpr = tpr_interp.mean(axis=0)
    mean_tpr[-1] = 1.0
    macro_auc = auc(fpr_grid, mean_tpr)
    ax.plot(fpr_grid, mean_tpr, color="black", lw=2.5, linestyle="--",
            label=f"Macro avg (AUC={macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves (one-vs-rest)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}  (macro AUC={macro_auc:.4f})")


def plot_confmat(cm: np.ndarray, labels: list[int], out_path: Path) -> None:
    """Row-normalized confusion matrix heatmap (rows sum to 1.0)."""
    n_classes = len(labels)
    names = class_names(n_classes)
    cm = np.asarray(cm, dtype=float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalized)")
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_per_class(per_class: dict, labels: list[int], out_path: Path) -> None:
    """Bar chart of precision / recall / F1 per class."""
    n_classes = len(labels)
    names = class_names(n_classes)
    precision = np.array(per_class["precision"])
    recall = np.array(per_class["recall"])
    f1 = np.array(per_class["f1"])

    x = np.arange(n_classes)
    w = 0.27
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w, precision, w, label="Precision", color="#4C78A8")
    ax.bar(x,     recall,    w, label="Recall",    color="#F58518")
    ax.bar(x + w, f1,        w, label="F1",        color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Per-class precision / recall / F1")
    ax.axhline(1 / n_classes, color="gray", lw=1, linestyle=":", label=f"chance ({1/n_classes:.2f})")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_per_subject(per_subject_acc: dict, out_path: Path) -> None:
    """Histogram of per-subject accuracies + summary stats."""
    accs = np.array(list(per_subject_acc.values()))
    mean = float(accs.mean())
    std = float(accs.std(ddof=1))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(accs, bins=20, color="#4C78A8", edgecolor="white")
    ax.axvline(mean, color="black", lw=2, linestyle="--",
               label=f"mean={mean:.3f}, std={std:.3f}, N={len(accs)}")
    ax.set_xlabel("Per-subject accuracy")
    ax.set_ylabel("Number of subjects")
    ax.set_title("Distribution of per-subject accuracies")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def main() -> None:
    args = parse_args()
    json_path = args.dir / f"{args.stem}.json"
    npz_path = args.dir / f"{args.stem}.npz"
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    summary = json.loads(json_path.read_text())
    data = np.load(npz_path)
    y_true = data["y_true"]
    y_prob = data["y_prob"]
    labels = data["labels"]

    plot_roc(y_true, y_prob, labels, args.dir / f"{args.stem}_roc.png")
    plot_confmat(
        np.array(summary["classification_report"]["confusion_matrix"]),
        summary["classification_report"]["labels"],
        args.dir / f"{args.stem}_confmat.png",
    )
    plot_per_class(
        summary["classification_report"]["per_class"],
        summary["classification_report"]["labels"],
        args.dir / f"{args.stem}_per_class.png",
    )
    plot_per_subject(
        summary["per_subject_acc"],
        args.dir / f"{args.stem}_per_subject.png",
    )


if __name__ == "__main__":
    main()
