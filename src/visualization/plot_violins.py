"""
plot_violins.py — Per-subject accuracy violin plots (LP / SFT / JADE).

Reads the same per-subject JSONs that statistical_tests.py consumes and writes
one violin figure per task to src/visualization/jade_<task>/figures/.

Each violin: KDE of per-subject accuracies. Overlay: one dot per subject,
black bar at the mean, dotted line at chance.

Usage:
    uv run python -m src.visualization.plot_violins
    uv run python -m src.visualization.plot_violins --methods sft jade
    uv run python -m src.visualization.plot_violins --paired-lines

Output: src/visualization/jade_9-class/figures/subject_violin.pdf
        src/visualization/jade_binary/figures/subject_violin.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Reuse the canonical run mapping from statistical_tests.py so the figure
# always shows the same configs as the headline statistics table.
from src.inference.statistical_tests import RUNS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "main-results"
FIGURES_ROOT = PROJECT_ROOT / "src" / "visualization"

TASKS = ["9-class", "binary"]
CHANCE = {"9-class": 1 / 9, "binary": 1 / 2}

COLORS = {
    "LP": "#9E9E9E",
    "SFT": "#4C78A8",
    "JADE": "#E45756",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--methods",
        nargs="+",
        choices=["lp", "sft", "jade"],
        default=["lp", "sft", "jade"],
        help="Which methods to include (default: all three).",
    )
    p.add_argument(
        "--paired-lines",
        action="store_true",
        help="Draw faint paired lines connecting the same subject across methods.",
    )
    p.add_argument(
        "--out-name",
        type=str,
        default="subject_violin",
        help="Output filename stem (default: subject_violin).",
    )
    return p.parse_args()


def load_per_subject(approach: str, task: str) -> dict[int, float]:
    """Load the per-subject accuracy dict from the winning-config JSON."""
    # statistical_tests.py uses key "ft" for SFT; CLI uses "sft" for clarity.
    key = "ft" if approach == "sft" else approach
    stem = RUNS[(key, task)]
    path = RESULTS_ROOT / f"{key}_{task}" / f"{stem}.json"
    data = json.loads(path.read_text())
    return {int(k): float(v) for k, v in data["per_subject_acc"].items()}


def panel_violin(
    ax: plt.Axes,
    accs_by_method: dict[str, np.ndarray],
    task: str,
    paired_lines: bool,
) -> None:
    """Draw one task panel: violins + strip + means + chance line."""
    methods = list(accs_by_method.keys())
    n_methods = len(methods)
    positions = np.arange(n_methods)

    parts = ax.violinplot(
        [accs_by_method[m] for m in methods],
        positions=positions,
        widths=0.7,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(COLORS[methods[i]])
        body.set_edgecolor(COLORS[methods[i]])
        body.set_alpha(0.35)

    rng = np.random.default_rng(0)
    for i, m in enumerate(methods):
        accs = accs_by_method[m]
        jitter = rng.uniform(-0.10, 0.10, size=len(accs))
        ax.scatter(
            positions[i] + jitter,
            accs,
            s=10,
            color=COLORS[m],
            edgecolor="white",
            linewidth=0.4,
            alpha=0.75,
            zorder=3,
        )

    for i, m in enumerate(methods):
        mean = float(accs_by_method[m].mean())
        ax.hlines(
            mean,
            positions[i] - 0.32,
            positions[i] + 0.32,
            colors="black",
            linewidth=1.8,
            zorder=4,
        )

    if paired_lines and n_methods >= 2:
        all_keys = set.intersection(*[set(range(len(accs_by_method[m]))) for m in methods])
        for sid in sorted(all_keys):
            ys = [accs_by_method[m][sid] for m in methods]
            ax.plot(
                positions,
                ys,
                color="gray",
                linewidth=0.3,
                alpha=0.25,
                zorder=1,
            )

    ax.axhline(
        CHANCE[task],
        color="gray",
        linestyle=":",
        linewidth=1.0,
        alpha=0.7,
        zorder=0,
    )
    ax.text(
        n_methods - 0.55,
        CHANCE[task] + 0.005,
        f"chance ({CHANCE[task]:.2f})",
        fontsize=8,
        color="gray",
        ha="right",
        va="bottom",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Per-subject accuracy")
    ax.set_title(f"{task} — N = {len(next(iter(accs_by_method.values())))} subjects")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def main() -> None:
    args = parse_args()

    label_map = {"lp": "LP", "sft": "SFT", "jade": "JADE"}

    for task in TASKS:
        per_method = {m: load_per_subject(m, task) for m in args.methods}
        common = sorted(set.intersection(*[set(d.keys()) for d in per_method.values()]))
        if not common:
            raise RuntimeError(f"No subjects shared across methods for task {task}")
        accs_by_method = {
            label_map[m]: np.array([per_method[m][sid] for sid in common], dtype=float)
            for m in args.methods
        }

        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        panel_violin(ax, accs_by_method, task, paired_lines=args.paired_lines)
        fig.tight_layout()

        out_dir = FIGURES_ROOT / f"jade_{task}" / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.out_name}.pdf"
        fig.savefig(out_path)
        print(f"Saved → {out_path.relative_to(PROJECT_ROOT)}")
        plt.close(fig)

        print(f"\n{task}:")
        for label, accs in accs_by_method.items():
            print(
                f"  {label:>4}: N={len(accs)}  "
                f"mean={accs.mean():.4f}  std={accs.std(ddof=1):.4f}  "
                f"min={accs.min():.4f}  max={accs.max():.4f}"
            )


if __name__ == "__main__":
    main()
