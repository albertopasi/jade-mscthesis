"""Paired per-subject accuracy scatter.

Default: JADE (y) vs SFT (x). Points above y=x are subjects helped by SupCon.
With --with-lp, also produces a side-by-side figure with JADE-vs-SFT (left)
and JADE-vs-LP (right), saved separately as _with_lp.pdf.

Usage:
    uv run python -m src.visualization.make_paired_scatter --task 9-class
    uv run python -m src.visualization.make_paired_scatter --task binary --with-lp
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from src.visualization._common import (
    COLOR_JADE, FT_RUNS, JADE_RUNS, LP_RUNS,
    LABEL_JADE, LABEL_LP, LABEL_SFT,
    PROJECT_ROOT, add_title_arg, load_run, save_fig, setup_style,
)


def _scatter_panel(ax, x: np.ndarray, y: np.ndarray, x_label: str, y_label: str) -> None:
    delta = y - x
    wins = int((delta > 0).sum())
    losses = int((delta < 0).sum())
    ties = int((delta == 0).sum())
    mean_delta = delta.mean()

    lo = min(x.min(), y.min()) - 3
    hi = max(x.max(), y.max()) + 3
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, linestyle="--", label="y = x")
    ax.scatter(x, y, s=24, color=COLOR_JADE, alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{x_label} subject accuracy (%)")
    ax.set_ylabel(f"{y_label} subject accuracy (%)")
    ax.grid(alpha=0.3, linestyle=":")
    txt = (
        f"n = {len(x)} subjects\n"
        f"{y_label} > {x_label}: {wins}\n"
        f"{y_label} < {x_label}: {losses}\n"
        f"Tie: {ties}\n"
        f"mean Δ = {mean_delta:+.2f} pp"
    )
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9))
    ax.legend(loc="lower right", framealpha=0.9)


def _paired(jade: dict[str, float], other: dict[str, float]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    common = sorted(set(jade) & set(other), key=int)
    if not common:
        raise SystemExit("No overlapping subjects between runs.")
    x = np.array([other[sid] for sid in common]) * 100
    y = np.array([jade[sid] for sid in common]) * 100
    return x, y, common


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["9-class", "binary"], required=True)
    ap.add_argument("--with-lp", action="store_true",
                    help="Also produce a 2-panel figure including JADE-vs-LP.")
    add_title_arg(ap)
    args = ap.parse_args()

    setup_style()
    jade_sum, _ = load_run("jade", args.task, JADE_RUNS[args.task])
    ft_sum, _ = load_run("ft", args.task, FT_RUNS[args.task])
    jade_acc = jade_sum["per_subject_acc"]
    ft_acc = ft_sum["per_subject_acc"]

    if not args.with_lp:
        x, y, _ = _paired(jade_acc, ft_acc)
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        _scatter_panel(ax, x, y, LABEL_SFT, LABEL_JADE)
        if args.title:
            ax.set_title(f"Paired per-subject comparison ({args.task})")
        out = PROJECT_ROOT / "src" / "visualization" / f"jade_{args.task}" / "figures" / "paired_subject_scatter.pdf"
        save_fig(fig, out)
        return

    # --with-lp: side-by-side panels (JADE vs SFT  |  JADE vs LP)
    lp_sum, _ = load_run("lp", args.task, LP_RUNS[args.task])
    lp_acc = lp_sum["per_subject_acc"]
    x_ft, y_ft, _ = _paired(jade_acc, ft_acc)
    x_lp, y_lp, _ = _paired(jade_acc, lp_acc)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5.5))
    _scatter_panel(axL, x_ft, y_ft, LABEL_SFT, LABEL_JADE)
    _scatter_panel(axR, x_lp, y_lp, LABEL_LP, LABEL_JADE)
    if args.title:
        fig.suptitle(f"Paired per-subject comparison ({args.task})")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()
    out = PROJECT_ROOT / "src" / "visualization" / f"jade_{args.task}" / "figures" / "paired_subject_scatter_with_lp.pdf"
    save_fig(fig, out)


if __name__ == "__main__":
    main()
