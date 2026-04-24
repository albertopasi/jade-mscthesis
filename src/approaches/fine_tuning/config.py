"""
config.py — Configuration and hyperparameters for Fine-Tuning with LoRA.

Two-stage pipeline: LP warmup (frozen encoder) → FT with LoRA adapters.
All defaults, paths, and hardware constants live here. The FTConfig dataclass
is populated from CLI arguments in train_ft.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.approaches.shared.config import (
    DATA_ROOTS,
    DATASET_DEFAULTS,
    DEVICE,
    NUM_WORKERS,
    PROJECT_ROOT,
    SAMPLING_RATE,
)

__all__ = [
    "DEVICE",
    "NUM_WORKERS",
    "SAMPLING_RATE",
]

# ── Paths ─────────────────────────────────────────────────────────────────────

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ft_checkpoints"

# ── W&B ───────────────────────────────────────────────────────────────────────

USE_WANDB = True
WANDB_PROJECT = "eeg-ft-v2"
WANDB_ENTITY = "zl-tudelft-thesis"


@dataclass
class FTConfig:
    """Mutable run configuration for two-stage fine-tuning."""

    # Dataset
    dataset: str = "faced"
    task_mode: str = "binary"
    fold: int | None = None
    generalization: bool = False
    gen_seeds: list[int] = field(default_factory=lambda: [123, 456, 789])

    # Window
    window_size: int = 2000  # 10 s at 200 Hz
    stride: int = 2000  # non-overlapping

    # Pooling
    pooling: str = "no"  # "no", "last", "last_avg"

    # Shared training
    batch_size: int = 64
    scale_factor: float = 1000.0
    use_mixup: bool = True
    use_amp: bool = True
    weight_decay: float = 0.01

    # ── LP warmup stage ────────────────────────────────────────────────
    lp_max_epochs: int = 20
    lp_lr: float = 5e-3
    lp_dropout: float = 0.05
    lp_warmup_epochs: int = 3
    lp_scheduler_patience: int = 6
    lp_early_stop_patience: int = 10
    lp_grad_clip: float = 2.0

    # ── FT stage ───────────────────────────────────────────────────────
    ft_max_epochs: int = 200
    ft_lr: float = 1e-4
    ft_dropout: float = 0.1
    ft_warmup_epochs: int = 5
    ft_scheduler_patience: int = 6
    ft_early_stop_patience: int = 20
    ft_grad_clip: float = 2.0

    # ── LoRA ───────────────────────────────────────────────────────────
    lora_rank: int = 16
    lora_alpha: int = 16  # match rank (official default)
    lora_dropout: float = 0.0  # official uses no LoRA dropout
    lora_target: str = "attention"  # "attention" only for now

    # ── Optional mode flags ───────────────────────────────────────────
    full_ft: bool = False  # True → full fine-tuning (no LoRA)
    reve_split: bool = False  # True → fixed train/val/test (FACED only)

    # ── Derived helpers ────────────────────────────────────────────────

    @property
    def data_root(self) -> Path:
        return DATA_ROOTS[self.dataset]

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
        return f"pool_{self.pooling}"

    @property
    def mixup_tag(self) -> str:
        return "" if self.use_mixup else "_nomixup"

    @property
    def _mode_tags(self) -> str:
        """Append '_fullft' and/or '_revesplit' labels when active."""
        tags = ""
        if self.full_ft:
            tags += "_fullft"
        if self.reve_split:
            tags += "_revesplit"
        return tags

    @property
    def _optim_tag(self) -> str:
        return f"b{self.batch_size}_lr{self.ft_lr:g}"

    def run_name(self, fold_idx: int, gen_seed: int | None = None) -> str:
        gen = f"_gen_s{gen_seed}" if gen_seed is not None else ""
        return (
            f"ft_{self.dataset}_{self.task_mode}_"
            f"{self.window_tag}_{self.pool_tag}_"
            f"r{self.lora_rank}_{self._optim_tag}{self.mixup_tag}{self._mode_tags}{gen}_fold_{fold_idx}"
        )

    def group_name(self) -> str:
        gen = "_gen" if self.generalization else ""
        return (
            f"ft_{self.dataset}_{self.task_mode}_"
            f"{self.window_tag}_{self.pool_tag}_"
            f"r{self.lora_rank}_{self._optim_tag}{self.mixup_tag}{self._mode_tags}{gen}"
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
            "task_mode": self.task_mode,
            "fold": fold_idx,
            "n_folds": n_folds,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "window_size": self.window_size,
            "stride": self.stride,
            "pooling": self.pooling,
            "n_train_subjects": n_train_subjects,
            "n_val_subjects": n_val_subjects,
            "n_train_windows": n_train_windows,
            "n_val_windows": n_val_windows,
            "embed_dim": embed_dim,
            "generalization": self.generalization,
            "gen_seed": gen_seed,
            "use_mixup": self.use_mixup,
            "use_amp": self.use_amp,
            "scale_factor": self.scale_factor,
            "full_ft": self.full_ft,
            "reve_split": self.reve_split,
            # LP stage
            "lp_max_epochs": self.lp_max_epochs,
            "lp_lr": self.lp_lr,
            "lp_dropout": self.lp_dropout,
            "lp_warmup_epochs": self.lp_warmup_epochs,
            "lp_scheduler_patience": self.lp_scheduler_patience,
            "lp_early_stop_patience": self.lp_early_stop_patience,
            "lp_grad_clip": self.lp_grad_clip,
            # FT stage
            "ft_max_epochs": self.ft_max_epochs,
            "ft_lr": self.ft_lr,
            "ft_dropout": self.ft_dropout,
            "ft_warmup_epochs": self.ft_warmup_epochs,
            "ft_scheduler_patience": self.ft_scheduler_patience,
            "ft_early_stop_patience": self.ft_early_stop_patience,
            "ft_grad_clip": self.ft_grad_clip,
            # LoRA
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target": self.lora_target,
        }
