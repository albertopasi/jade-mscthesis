"""Per-class F1 bar chart: JADE vs SFT (and optionally LP).

Shows which emotions benefit most from SupCon. With --with-lp, adds a third
bar per class for the LP baseline (writes a separate _with_lp.pdf so the
two-way comparison plot is preserved).

Usage:
    uv run python -m src.visualization.make_per_class_f1_bars --task 9-class
    uv run python -m src.visualization.make_per_class_f1_bars --task binary --with-lp
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from src.visualization._common import (
    COLOR_FT,
    COLOR_JADE,
    COLOR_LP,
    FT_RUNS,
    JADE_RUNS,
    LABEL_JADE,
    LABEL_LP,
    LABEL_SFT,
    LP_RUNS,
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
    ap.add_argument(
        "--with-lp", action="store_true", help="Add LP baseline as a third bar per class."
    )
    add_title_arg(ap)
    args = ap.parse_args()

    setup_style()
    names = class_names(args.task)

    jade_sum, _ = load_run("jade", args.task, JADE_RUNS[args.task])
    ft_sum, _ = load_run("ft", args.task, FT_RUNS[args.task])
    lp_sum = load_run("lp", args.task, LP_RUNS[args.task])[0] if args.with_lp else None

    f1_jade = np.array(jade_sum["classification_report"]["per_class"]["f1"])
    f1_ft = np.array(ft_sum["classification_report"]["per_class"]["f1"])
    macro_jade = jade_sum["classification_report"]["macro"]["f1"]
    macro_ft = ft_sum["classification_report"]["macro"]["f1"]

    if lp_sum is not None:
        f1_lp = np.array(lp_sum["classification_report"]["per_class"]["f1"])
        macro_lp = lp_sum["classification_report"]["macro"]["f1"]

    delta = f1_jade - f1_ft  # JADE vs SFT delta (always shown — the main story)
    macro_delta = macro_jade - macro_ft

    n = len(names)
    x = np.arange(n)
    macro_x = n + 0.7

    if args.with_lp:
        width = 0.27
        offsets = [-width, 0.0, width]  # LP, SFT, JADE
    else:
        width = 0.38
        offsets = [-width / 2, width / 2]  # SFT, JADE

    fig, ax = plt.subplots(figsize=(max(7.0, 0.9 * n + 3), 4.4))

    def _draw(xs, vals, off, color, label=None):
        ax.bar(xs + off, vals, width, label=label, color=color, edgecolor="black", linewidth=0.4)

    if args.with_lp:
        _draw(x, f1_lp, offsets[0], COLOR_LP, label=LABEL_LP)
        _draw(x, f1_ft, offsets[1], COLOR_FT, label=LABEL_SFT)
        _draw(x, f1_jade, offsets[2], COLOR_JADE, label=LABEL_JADE)
        _draw(np.array([macro_x]), [macro_lp], offsets[0], COLOR_LP)
        _draw(np.array([macro_x]), [macro_ft], offsets[1], COLOR_FT)
        _draw(np.array([macro_x]), [macro_jade], offsets[2], COLOR_JADE)
    else:
        _draw(x, f1_ft, offsets[0], COLOR_FT, label=LABEL_SFT)
        _draw(x, f1_jade, offsets[1], COLOR_JADE, label=LABEL_JADE)
        _draw(np.array([macro_x]), [macro_ft], offsets[0], COLOR_FT)
        _draw(np.array([macro_x]), [macro_jade], offsets[1], COLOR_JADE)

    ax.axvline(n - 0.15, color="#999999", linewidth=0.7, linestyle="--")

    # Delta annotations (JADE vs SFT) above each group
    ymax_vals = [f1_jade.max(), f1_ft.max(), macro_jade, macro_ft]
    if args.with_lp:
        ymax_vals += [f1_lp.max(), macro_lp]
    ymax = max(ymax_vals)
    for xi, d in list(zip(x, delta)) + [(macro_x, macro_delta)]:
        sign = "+" if d >= 0 else ""
        color = "#2ca02c" if d >= 0 else "#d62728"
        ax.text(
            xi,
            ymax + 0.04,
            f"{sign}{d * 100:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
            fontweight="bold",
        )

    ax.set_xticks(list(x) + [macro_x])
    ax.set_xticklabels(names + ["Macro avg"], rotation=30, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0, min(1.0, ymax + 0.15))
    if args.title:
        suffix = " + LP" if args.with_lp else ""
        ax.set_title(f"Per-class F1: JADE vs SFT{suffix} ({args.task})  —  Δ (JADE − SFT) above")
    # Legend ordered top→bottom: JADE, SFT, (LP)
    handles, labels = ax.get_legend_handles_labels()
    desired = [LABEL_JADE, LABEL_SFT, LABEL_LP]
    order = [labels.index(d) for d in desired if d in labels]
    ax.legend(
        [handles[i] for i in order], [labels[i] for i in order], loc="lower right", framealpha=0.9
    )
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    fname = "per_class_f1_bars_with_lp.pdf" if args.with_lp else "per_class_f1_bars.pdf"
    out = PROJECT_ROOT / "src" / "visualization" / f"jade_{args.task}" / "figures" / fname
    save_fig(fig, out)


if __name__ == "__main__":
    main()
