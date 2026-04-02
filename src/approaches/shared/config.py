"""
config.py — Shared constants and paths used across all approaches.

Imported by both LP and FT pipelines. Approach-specific config
(output dirs, W&B settings, LPConfig/FTConfig) lives in each approach's
own config.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ── Data paths ────────────────────────────────────────────────────────────────

DATA_ROOTS = {
    "thu-ep": PROJECT_ROOT / "data" / "thu ep" / "preprocessed_v2",
    "faced": PROJECT_ROOT / "data" / "FACED" / "preprocessed_v2",
}

# ── REVE model paths ──────────────────────────────────────────────────────────

REVE_MODEL_PATH = PROJECT_ROOT / "models" / "reve_pretrained_original" / "reve-base"
REVE_POS_PATH = PROJECT_ROOT / "models" / "reve_pretrained_original" / "reve-positions"

# ── Hardware ──────────────────────────────────────────────────────────────────

SAMPLING_RATE = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0 if sys.platform == "win32" else 4

# ── Dataset-specific defaults ─────────────────────────────────────────────────

DATASET_DEFAULTS = {
    "thu-ep": {
        "n_channels": 30,
        "scale_factor": 1000.0,
    },
    "faced": {
        "n_channels": 32,
        "scale_factor": 1000.0,
    },
}
