"""
dataset.py — Shared dataset factory for REVE-based pipelines.

Used by both LP and FT training scripts. The cfg argument is duck-typed:
any config with .dataset, .task_mode, .data_root, .window_size, .stride,
.scale_factor fields works (LPConfig and FTConfig both qualify).
"""

from __future__ import annotations


def build_raw_dataset(
    cfg,
    subject_ids: list[int],
    stimulus_filter: set[int] | None = None,
):
    """Build a raw EEG window dataset for the given subjects and config."""
    if cfg.dataset == "faced":
        from src.datasets.faced_dataset import FACEDWindowDataset
        return FACEDWindowDataset(
            subject_ids=subject_ids,
            task_mode=cfg.task_mode,
            data_root=cfg.data_root,
            window_size=cfg.window_size,
            stride=cfg.stride,
            scale_factor=cfg.scale_factor,
            stimulus_filter=stimulus_filter,
        )
    else:
        from src.datasets.thu_ep_dataset import THUEPWindowDataset
        return THUEPWindowDataset(
            subject_ids=subject_ids,
            task_mode=cfg.task_mode,
            data_root=cfg.data_root,
            window_size=cfg.window_size,
            stride=cfg.stride,
            scale_factor=cfg.scale_factor,
            stimulus_filter=stimulus_filter,
        )
