"""
summary.py — Cross-fold and cross-seed summary for JADE.

Thin wrappers around the shared summary module.
"""

from __future__ import annotations

from src.approaches.jade.config import OUTPUT_DIR, JADEConfig
from src.approaches.shared.summary import (
    print_cross_seed_summary as _print_cross_seed_summary,
)
from src.approaches.shared.summary import (
    print_fold_summary as _print_fold_summary,
)


def _fold_filename(cfg: JADEConfig, gen_seed: int | None) -> str:
    gen_tag = f"_gen_s{gen_seed}" if gen_seed is not None else ""
    return (
        f"summary_{cfg.dataset}_{cfg.task_mode}_{cfg.window_tag}_"
        f"{cfg.pool_tag}_r{cfg.lora_rank}_{cfg._supcon_tag}"
        f"{cfg.mixup_tag}{cfg._mode_tags}{gen_tag}"
    )


def _cross_seed_filename(cfg: JADEConfig) -> str:
    return (
        f"summary_{cfg.dataset}_{cfg.task_mode}_{cfg.window_tag}_"
        f"{cfg.pool_tag}_r{cfg.lora_rank}_{cfg._supcon_tag}"
        f"{cfg.mixup_tag}{cfg._mode_tags}_gen_cross_seed"
    )


def _approach_label(cfg: JADEConfig) -> str:
    return f"JADE a={cfg.supcon_alpha} r={cfg.lora_rank}"


def print_fold_summary(
    cfg: JADEConfig,
    fold_results: list[dict],
    gen_seed: int | None = None,
) -> None:
    _print_fold_summary(
        cfg,
        fold_results,
        output_dir=OUTPUT_DIR,
        approach_label=_approach_label(cfg),
        filename_stem=_fold_filename(cfg, gen_seed),
        extra_json={
            "approach": "jade_fullft" if cfg.full_ft else "jade_lora",
            "lora_rank": cfg.lora_rank,
            "supcon_alpha": cfg.supcon_alpha,
            "supcon_temperature": cfg.supcon_temperature,
            "supcon_repr": cfg.supcon_repr,
        },
        gen_seed=gen_seed,
    )


def print_cross_seed_summary(
    cfg: JADEConfig,
    seed_summaries: list[dict],
) -> None:
    _print_cross_seed_summary(
        cfg,
        seed_summaries,
        output_dir=OUTPUT_DIR,
        approach_label=_approach_label(cfg),
        filename_stem=_cross_seed_filename(cfg),
        extra_json={
            "approach": "jade_fullft" if cfg.full_ft else "jade_lora",
            "lora_rank": cfg.lora_rank,
            "supcon_alpha": cfg.supcon_alpha,
            "supcon_temperature": cfg.supcon_temperature,
            "supcon_repr": cfg.supcon_repr,
        },
    )
