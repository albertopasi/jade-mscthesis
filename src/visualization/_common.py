"""Shared utilities for visualization scripts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "main-results"

CLASS_NAMES_9 = [
    "Anger",
    "Disgust",
    "Fear",
    "Sadness",
    "Neutral",
    "Amusement",
    "Inspiration",
    "Joy",
    "Tenderness",
]
CLASS_NAMES_2 = ["Negative", "Positive"]

CHANCE = {"9-class": 1 / 9, "binary": 0.5}

COLOR_JADE = "#1f77b4"
COLOR_FT = "#d62728"  # SFT
COLOR_LP = "#ff7f0e"
COLOR_MEAN = "#2ca02c"
COLOR_CHANCE = "#7f7f7f"
COLOR_BAR_LIGHT = "#b8b8b8"

LABEL_JADE = "JADE"
LABEL_SFT = "SFT"
LABEL_LP = "LP"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def class_names(task: str) -> list[str]:
    return CLASS_NAMES_9 if task == "9-class" else CLASS_NAMES_2


def add_title_arg(ap) -> None:
    """Add --title / --no-title to a parser. Default: no title (publication style)."""
    ap.add_argument("--title", dest="title", action="store_true", help="Show figure title.")
    ap.add_argument(
        "--no-title", dest="title", action="store_false", help="Hide figure title (default)."
    )
    ap.set_defaults(title=False)


def load_run(approach: str, task: str, stem: str) -> tuple[dict, dict]:
    """Return (json_summary, npz_arrays_dict) for one run."""
    sub = RESULTS_ROOT / f"{approach}_{task}"
    j = json.loads((sub / f"{stem}.json").read_text())
    npz = np.load(sub / f"{stem}.npz")
    arrays = {k: npz[k] for k in npz.files}
    return j, arrays


def save_fig(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path.relative_to(PROJECT_ROOT)}")


JADE_RUNS = {
    "9-class": "jade_faced_9-class_w10s10_pool_no_r16_a0.3_t0.2_context_b256_lr0.0004_fullft",
    "binary": "jade_faced_binary_w10s10_pool_no_r16_a0.2_t0.05_context_b128_lr0.0001_fullft",
}
FT_RUNS = {
    "9-class": "ft_faced_9-class_w10s10_pool_no_r16_b256_lr0.0004_nomixup_fullft",
    "binary": "ft_faced_binary_w10s10_pool_no_r16_nomixup_fullft",
}
LP_RUNS = {
    "9-class": "lp_faced_v2_9-class_w10s10_pool_no_official_nomixup",
    "binary": "lp_faced_v2_binary_w10s10_pool_no_official_nomixup",
}
