"""
config.py — Configuration and hyperparameters for Linear Probing.

All defaults, paths, and hardware constants live here. The LPConfig dataclass
is populated from CLI arguments in train_lp.py and passed to run_fold()
and summary functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from src.approaches.shared.config import (
    DATA_ROOTS,
    DATASET_DEFAULTS,
    PROJECT_ROOT,
    SAMPLING_RATE,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

EMBEDDINGS_ROOTS = {
    "thu-ep": PROJECT_ROOT / "data" / "thu ep" / "embeddings_v2",
    "faced": PROJECT_ROOT / "data" / "FACED" / "embeddings_v2",
}

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "lp_checkpoints"

# ── W&B ───────────────────────────────────────────────────────────────────────

USE_WANDB = True
WANDB_PROJECT = "eeg-lp-v2"
WANDB_ENTITY = "zl-tudelft-thesis"

# ── Hardware (LP-specific) ────────────────────────────────────────────────────

ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"


@dataclass
class LPConfig:
    """Mutable run configuration populated from CLI arguments."""

    # Data version
    data_tag: str = "v2"

    # Dataset
    dataset: str = "faced"

    # Mode
    official_mode: bool = True  # True = frozen encoder + trainable query token
    # False = pre-computed embeddings (fast mode)

    # Task
    task_mode: str = "binary"
    fold: int | None = None
    generalization: bool = False
    gen_seeds: list[int] = field(default_factory=lambda: [123, 456, 789])

    # Window
    window_size: int = 2000  # 10 s at 200 Hz
    stride: int = 2000  # non-overlapping

    # Pooling (official mode)
    pooling: str = "no"  # "no", "last", "last_avg"

    # Training
    max_epochs: int = 50
    batch_size: int = 64
    lr: float = 5e-3
    dropout: float = 0.05
    weight_decay: float = 0.01
    warmup_epochs: int = 3
    scheduler_patience: int = 5
    early_stop_patience: int = 11
    grad_clip: float = 100.0

    # Regularization / precision
    use_mixup: bool = True
    use_amp: bool = True

    # Features (fast mode only)
    normalize_features: bool = False
    use_pooling: bool = True  # fast-mode pooling toggle
    no_pool_mode: str = "mean"  # fast-mode: "mean" or "flat"

    # Scale factor
    scale_factor: float = 1000.0

    # ── Derived helpers ───────────────────────────────────────────────────

    @property
    def data_root(self) -> Path:
        return DATA_ROOTS[self.dataset]

    @property
    def embeddings_dir(self) -> Path:
        return EMBEDDINGS_ROOTS[self.dataset]

    @property
    def n_channels(self) -> int:
        return DATASET_DEFAULTS[self.dataset]["n_channels"]

    @property
    def num_classes(self) -> int:
        return 2 if self.task_mode == "binary" else 9

    @property
    def window_tag(self) -> str:
        w_s = round(self.window_size / SAMPLING_RATE)
        st_s = round(self.stride / SAMPLING_RATE)
        return f"w{w_s}s{st_s}"

    @property
    def pool_tag(self) -> str:
        if self.official_mode:
            return f"pool_{self.pooling}"
        return "pool" if self.use_pooling else f"nopool_{self.no_pool_mode}"

    @property
    def mode_tag(self) -> str:
        return "official" if self.official_mode else "fast"

    @property
    def norm_tag(self) -> str:
        return "_norm" if self.normalize_features else ""

    @property
    def mixup_tag(self) -> str:
        return "" if self.use_mixup else "_nomixup"

    def run_name(self, fold_idx: int, gen_seed: int | None = None) -> str:
        gen = f"_gen_s{gen_seed}" if gen_seed is not None else ""
        return (
            f"lp_{self.dataset}_{self.data_tag}_{self.task_mode}_"
            f"{self.window_tag}_{self.pool_tag}_{self.mode_tag}"
            f"{self.norm_tag}{self.mixup_tag}{gen}_fold_{fold_idx}"
        )

    def group_name(self) -> str:
        gen = "_gen" if self.generalization else ""
        return (
            f"lp_{self.dataset}_{self.data_tag}_{self.task_mode}_"
            f"{self.window_tag}_{self.pool_tag}_{self.mode_tag}"
            f"{self.norm_tag}{self.mixup_tag}{gen}"
        )

    def hparams_dict(
        self,
        fold_idx: int,
        n_folds: int,
        n_train_subjects: int,
        n_val_subjects: int,
        n_train_windows: int,
        n_val_windows: int,
        embed_dim: int,
        gen_seed: int | None = None,
    ) -> dict:
        return {
            "dataset": self.dataset,
            "data_tag": self.data_tag,
            "official_mode": self.official_mode,
            "task_mode": self.task_mode,
            "fold": fold_idx,
            "n_folds": n_folds,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "dropout": self.dropout,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "scheduler_patience": self.scheduler_patience,
            "early_stop_patience": self.early_stop_patience,
            "max_epochs": self.max_epochs,
            "window_size": self.window_size,
            "stride": self.stride,
            "pooling": self.pooling if self.official_mode else self.pool_tag,
            "n_train_subjects": n_train_subjects,
            "n_val_subjects": n_val_subjects,
            "n_train_windows": n_train_windows,
            "n_val_windows": n_val_windows,
            "normalize_features": self.normalize_features,
            "embed_dim": embed_dim,
            "generalization": self.generalization,
            "gen_seed": gen_seed,
            "use_mixup": self.use_mixup,
            "use_amp": self.use_amp,
            "grad_clip": self.grad_clip,
            "scale_factor": self.scale_factor,
        }
