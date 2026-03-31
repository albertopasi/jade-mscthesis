"""
summary.py — Cross-fold and cross-seed summary for FT.

Thin wrappers around the shared summary module.
"""

from __future__ import annotations

from src.approaches.fine_tuning.config import FTConfig, OUTPUT_DIR
from src.approaches.shared.summary import (
    print_fold_summary as _print_fold_summary,
    print_cross_seed_summary as _print_cross_seed_summary,
)


def _fold_filename(cfg: FTConfig, gen_seed: int | None) -> str:
    gen_tag = f"_gen_s{gen_seed}" if gen_seed is not None else ""
    return (
        f"summary_{cfg.dataset}_{cfg.task_mode}_{cfg.window_tag}_"
        f"{cfg.pool_tag}_r{cfg.lora_rank}{cfg.mixup_tag}{cfg._mode_tags}{gen_tag}"
    )


def _cross_seed_filename(cfg: FTConfig) -> str:
    return (
        f"summary_{cfg.dataset}_{cfg.task_mode}_{cfg.window_tag}_"
        f"{cfg.pool_tag}_r{cfg.lora_rank}{cfg.mixup_tag}{cfg._mode_tags}_gen_cross_seed"
    )


def print_fold_summary(
    cfg: FTConfig,
    fold_results: list[dict],
    gen_seed: int | None = None,
) -> None:
    _print_fold_summary(
        cfg,
        fold_results,
        output_dir=OUTPUT_DIR,
        approach_label=f"LoRA r={cfg.lora_rank}",
        filename_stem=_fold_filename(cfg, gen_seed),
        extra_json={"approach": "ft_lora", "lora_rank": cfg.lora_rank},
        gen_seed=gen_seed,
    )


def print_cross_seed_summary(
    cfg: FTConfig,
    seed_summaries: list[dict],
) -> None:
    _print_cross_seed_summary(
        cfg,
        seed_summaries,
        output_dir=OUTPUT_DIR,
        approach_label=f"LoRA r={cfg.lora_rank}",
        filename_stem=_cross_seed_filename(cfg),
        extra_json={"approach": "ft_lora", "lora_rank": cfg.lora_rank},
    )
