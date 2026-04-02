"""
training_utils.py — Shared training loop utilities.

Used by both LP and FT training loops.
"""

from __future__ import annotations

COL_W = 105


def fmt_dur(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _get_exponential_warmup_lambda(total_steps: int):
    """Exponential warmup schedule matching official REVE."""

    def fn(step: int) -> float:
        if step >= total_steps or total_steps == 0:
            return 1.0
        return min(1.0, (10 ** (step / total_steps) - 1) / 9)

    return fn
