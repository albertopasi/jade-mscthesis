"""Row-normalized confusion matrix for JADE (per-class recall on the diagonal).

Usage:
    uv run python -m src.visualization.make_confusion_matrix --task 9-class
    uv run python -m src.visualization.make_confusion_matrix --task binary
"""

from __future__ import annotations

import argparse

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.visualization._common import (
    JADE_RUNS,
    PROJECT_ROOT,
    add_title_arg,
    class_names,
    load_run,
    save_fig,
    setup_style,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["9-class", "binary"], required=True)
    add_title_arg(ap)
    args = ap.parse_args()

    setup_style()
    stem = JADE_RUNS[args.task]
    _, arr = load_run("jade", args.task, stem)
    names = class_names(args.task)
    labels = list(range(len(names)))

    cm = confusion_matrix(arr["y_true"], arr["y_pred"], labels=labels)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100  # row-normalized, percent

    n = len(names)
    fig, ax = plt.subplots(figsize=(max(5.5, 0.65 * n + 2), max(5.0, 0.65 * n + 1.5)))
    # Power-norm (gamma<1) gives more color resolution to small percentages,
    # so 5% vs 15% off-diagonal cells look visually distinct.
    norm = mcolors.PowerNorm(gamma=0.55, vmin=0, vmax=100)
    im = ax.imshow(cm_pct, cmap="Blues", norm=norm)

    # Threshold for switching annotation text to white: use the normalized
    # value (so it scales with the colormap, not the raw percentage).
    for i in range(n):
        for j in range(n):
            v = cm_pct[i, j]
            color = "white" if norm(v) > 0.55 else "black"
            ax.text(
                j, i, f"{v:.1f}", ha="center", va="center", fontsize=9 if n > 5 else 12, color=color
            )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    if args.title:
        ax.set_title(f"JADE — confusion matrix ({args.task}, row-normalized)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out = (
        PROJECT_ROOT
        / "src"
        / "visualization"
        / f"jade_{args.task}"
        / "figures"
        / "confusion_matrix.pdf"
    )
    save_fig(fig, out)


if __name__ == "__main__":
    main()
