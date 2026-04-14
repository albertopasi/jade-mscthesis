"""
lora.py — LoRA configuration and adapter wrapping utilities.

Applies low-rank adapters to REVE transformer attention layers using the
peft library. Targets to_qkv and to_out in all 22 transformer layers.
"""

from __future__ import annotations

import torch.nn as nn
from peft import LoraConfig, get_peft_model


def get_lora_config(
    encoder: nn.Module,
    rank: int,
    alpha: int,
    dropout: float = 0.0,
    target: str = "attention",
) -> LoraConfig:
    """Build LoRA config targeting attention layers in all transformer layers.

    Args:
        encoder: The REVE encoder model.
        rank: LoRA rank (r).
        alpha: LoRA alpha scaling factor.
        dropout: LoRA dropout rate.
        target: Which modules to target ("attention" for now).

    Returns:
        LoraConfig ready for get_peft_model().
    """
    n_layers = len(encoder.transformer.layers)
    target_modules = []

    for i in range(n_layers):
        if "attention" in target:
            target_modules.append(f"transformer.layers.{i}.0.to_qkv")
            target_modules.append(f"transformer.layers.{i}.0.to_out")
        if "ffn" in target:
            target_modules.append(f"transformer.layers.{i}.1.net.1")
            target_modules.append(f"transformer.layers.{i}.1.net.3")

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
    )


def apply_lora(model: nn.Module, cfg) -> nn.Module:
    """Wrap the encoder with LoRA adapters.

    Args:
        model: Classifier with a .reve encoder attribute (FT or JADE).
        cfg: Config with lora_rank, lora_alpha, lora_dropout, lora_target attributes.

    Returns:
        The same model with model.reve replaced by a PeftModel.
    """
    config = get_lora_config(
        model.reve,
        rank=cfg.lora_rank,
        alpha=cfg.lora_alpha,
        dropout=cfg.lora_dropout,
        target=cfg.lora_target,
    )
    model.reve = get_peft_model(model.reve, config)
    return model


def print_lora_summary(model: nn.Module) -> None:
    """Print trainable vs total parameter counts after LoRA is applied."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(
        p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora_" in n
    )
    print(
        f"  Parameters: {trainable:,} trainable / {total:,} total "
        f"({100 * trainable / total:.2f}%)  |  LoRA: {lora_params:,}"
    )
