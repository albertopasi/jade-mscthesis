"""
training_utils.py — Shared training loop utilities.

Used by both LP and FT training loops.
"""

from __future__ import annotations

COL_W = 105


class _PatienceMonitor:
    """Early stopping monitor: stops when counter >= patience."""

    def __init__(self, patience: int = 10):
        self.patience = patience
        self.best_val = 0.0
        self.counter = 0

    def __call__(self, val: float) -> bool:
        if val > self.best_val:
            self.best_val = val
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


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
