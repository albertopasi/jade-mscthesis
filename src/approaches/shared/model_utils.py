"""
model_utils.py — Shared model building blocks for REVE-based classifiers.

Used by both LP and FT model definitions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (matching REVE backbone's RMSNorm)."""

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.scale


def compute_n_patches(window_size: int, patch_size: int = 200, overlap: int = 20) -> int:
    """Compute number of time patches from REVE's unfold parameters."""
    step = patch_size - overlap
    return (window_size - patch_size) // step + 1
