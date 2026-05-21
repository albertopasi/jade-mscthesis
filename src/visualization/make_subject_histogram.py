"""Subject-ordered accuracy histogram.

Each bar is one subject's accuracy on the val fold containing them.
Subjects sorted ascending. Adds horizontal lines for chance and the
cross-subject mean, plus a side bar showing mean +/- std.

Usage:
    uv run python -m src.visualization.make_subject_histogram --task 9-class
    uv run python -m src.visualization.make_subject_histogram --task binary
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from src.visualization._common import (
    CHANCE, COLOR_BAR_LIGHT, COLOR_CHANCE, COLOR_JADE, COLOR_MEAN, JADE_RUNS,
    PROJECT_ROOT, add_title_arg, load_run, save_fig, setup_style,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["9-class", "binary"], required=True)
    add_title_arg(ap)
    args = ap.parse_args()

    setup_style()
    stem = JADE_RUNS[args.task]
    summary, _ = load_run("jade", args.task, stem)

    accs = np.array([v for v in summary["per_subject_acc"].values()], dtype=float)
    accs_sorted = np.sort(accs)
    n = len(accs_sorted)
    mean = accs_sorted.mean()
    std = accs_sorted.std(ddof=1)
    chance = CHANCE[args.task]

    fig, (ax, ax_box) = plt.subplots(
        1, 2, figsize=(10, 4),
        gridspec_kw={"width_ratios": [12, 1], "wspace": 0.02},
        sharey=True,
    )

    x = np.arange(n)
    ax.bar(x, accs_sorted * 100, width=0.85, color=COLOR_BAR_LIGHT,
           edgecolor="#666666", linewidth=0.3, label="Per-subject accuracy")
    ax.axhline(mean * 100, color=COLOR_MEAN, linestyle="--", linewidth=1.2,
               label=f"Mean = {mean*100:.1f}%")
    ax.axhline(chance * 100, color="#d62728", linestyle=":", linewidth=1.2,
               label=f"Chance = {chance*100:.1f}%")

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 100)
    ax.set_xlabel(f"Subjects (sorted by accuracy, n = {n})")
    ax.set_ylabel("Accuracy (%)")
    if args.title:
        ax.set_title(f"JADE — per-subject accuracy ({args.task})")
    ax.legend(loc="upper left", framealpha=0.9)

    # Side "averaged accuracy" bar with std error bar
    ax_box.bar([0], [mean * 100], width=0.6, color=COLOR_JADE,
               edgecolor="black", linewidth=0.5)
    ax_box.errorbar([0], [mean * 100], yerr=[std * 100], fmt="none",
                    ecolor="black", capsize=4, linewidth=1)
    ax_box.set_xticks([0])
    ax_box.set_xticklabels(["mean ± std"], rotation=0)
    ax_box.set_xlim(-0.6, 0.6)
    ax_box.tick_params(axis="y", left=False, labelleft=False)
    ax_box.spines["left"].set_visible(False)

    out = PROJECT_ROOT / "src" / "visualization" / f"jade_{args.task}" / "figures" / "subject_histogram.pdf"
    save_fig(fig, out)


if __name__ == "__main__":
    main()
