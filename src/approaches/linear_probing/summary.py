"""
summary.py — Cross-fold and cross-seed summary for LP.

Thin wrappers around the shared summary module.
"""

from __future__ import annotations

from src.approaches.linear_probing.config import OUTPUT_DIR, LPConfig
from src.approaches.shared.summary import (
    fmt_metric,
)
from src.approaches.shared.summary import (
    print_cross_seed_summary as _print_cross_seed_summary,
)
from src.approaches.shared.summary import (
    print_fold_summary as _print_fold_summary,
)

# Re-export for backward compatibility
__all__ = ["print_fold_summary", "print_cross_seed_summary", "fmt_metric"]


def _fold_filename(cfg: LPConfig, gen_seed: int | None) -> str:
    gen_tag = f"_gen_s{gen_seed}" if gen_seed is not None else ""
    return (
        f"summary_{cfg.dataset}_{cfg.task_mode}_{cfg.window_tag}_"
        f"{cfg.pool_tag}_{cfg.mode_tag}{cfg.mixup_tag}{gen_tag}"
    )


def _cross_seed_filename(cfg: LPConfig) -> str:
    return (
        f"summary_{cfg.dataset}_{cfg.task_mode}_{cfg.window_tag}_"
        f"{cfg.pool_tag}_{cfg.mode_tag}{cfg.mixup_tag}_gen_cross_seed"
    )


def print_fold_summary(
    cfg: LPConfig,
    fold_results: list[dict],
    gen_seed: int | None = None,
) -> None:
    _print_fold_summary(
        cfg,
        fold_results,
        output_dir=OUTPUT_DIR,
        approach_label=cfg.mode_tag,
        filename_stem=_fold_filename(cfg, gen_seed),
        extra_json={"mode": cfg.mode_tag},
        gen_seed=gen_seed,
    )


def print_cross_seed_summary(
    cfg: LPConfig,
    seed_summaries: list[dict],
) -> None:
    _print_cross_seed_summary(
        cfg,
        seed_summaries,
        output_dir=OUTPUT_DIR,
        approach_label=cfg.mode_tag,
        filename_stem=_cross_seed_filename(cfg),
        extra_json={"mode": cfg.mode_tag},
    )
