"""
reve.py — Shared REVE model loading utilities.

Used by all approaches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel

from src.approaches.shared.config import REVE_MODEL_PATH, REVE_POS_PATH


def get_channel_names(dataset: str) -> list[str]:
    """Return electrode channel names for the specified dataset."""
    if dataset == "faced":
        from src.datasets.faced_dataset import FACED_CHANNELS
        return FACED_CHANNELS
    else:
        from src.preprocessing.thu_ep.config import THUEPConfig
        return THUEPConfig().final_channels


def load_reve_and_positions(
    channel_names: list[str],
    device: str = "cuda",
    reve_model_path: Path = REVE_MODEL_PATH,
    reve_pos_path: Path = REVE_POS_PATH,
) -> Tuple[nn.Module, Tensor]:
    """Load frozen REVE encoder and compute electrode position tensor.

    Returns:
        reve_model: The pretrained REVE model (eval mode, all params frozen).
        pos_tensor: Electrode positions, shape (n_channels, 3).
    """
    print("Loading REVE model from local path …")
    reve_model = AutoModel.from_pretrained(
        str(reve_model_path), trust_remote_code=True, torch_dtype="auto",
    )
    reve_model.eval()
    reve_model.to(device)
    for param in reve_model.parameters():
        param.requires_grad_(False)

    print("Loading REVE position bank from local path …")
    pos_bank = AutoModel.from_pretrained(
        str(reve_pos_path), trust_remote_code=True, torch_dtype="auto",
    )
    pos_bank.eval()
    pos_bank.to(device)

    with torch.no_grad():
        pos_tensor = pos_bank(channel_names)  # (n_channels, 3)
    print(f"  Electrode positions cached for {len(channel_names)} channels.")

    # Free position bank memory
    del pos_bank
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return reve_model, pos_tensor
