"""
model.py — REVE classifier for Fine-Tuning with LoRA.

Provides ReveClassifierFT: same architecture as ReveClassifierLP but supports
both frozen (LP stage) and unfrozen (FT stage) encoder modes. Gradients flow
through the encoder when unfrozen — controlled by requires_grad flags, not
torch.no_grad().
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.approaches.shared.model_utils import RMSNorm, compute_n_patches


class ReveClassifierFT(nn.Module):
    """
    Two-stage classifier: LP warmup + LoRA fine-tuning.

    Same architecture as ReveClassifierLP but:
    - forward() does NOT use torch.no_grad() on encoder
    - Supports freeze/unfreeze of encoder
    - Dropout is reconfigurable between stages
    """

    def __init__(
        self,
        reve_model: nn.Module,
        pos_tensor: Tensor,
        n_classes: int,
        n_channels: int,
        window_size: int,
        pooling: str = "no",
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        assert pooling in ("no", "last", "last_avg"), f"Pooling '{pooling}' not supported"

        self.pooling = pooling
        self.reve = reve_model
        self.embed_dim = reve_model.config.embed_dim  # 512 (Base)

        # Register electrode positions as buffer
        self.register_buffer("pos_tensor", pos_tensor)  # (n_channels, 3)

        # Trainable query token — initialised from pretrained checkpoint (matches official)
        self.cls_query_token = nn.Parameter(reve_model.cls_query_token.data.clone())

        # Compute dimensions
        n_patches = compute_n_patches(window_size)
        n_tokens = n_channels * n_patches

        if pooling == "no":
            out_shape = (1 + n_tokens) * self.embed_dim
        else:
            out_shape = self.embed_dim

        self.linear_head = nn.Sequential(
            RMSNorm(out_shape),
            nn.Dropout(dropout),
            nn.Linear(out_shape, n_classes),
        )
        self._out_shape = out_shape

        self.n_tokens = n_tokens
        self.n_channels = n_channels
        self.n_patches = n_patches

    def forward(self, eeg: Tensor, pos: Tensor | None = None) -> Tensor:
        """
        Args:
            eeg: (B, C, T) raw EEG.
            pos: (B, C, 3) electrode positions. If None, uses registered buffer.

        Returns:
            logits: (B, n_classes)
        """
        B = eeg.shape[0]

        if pos is None:
            pos = self.pos_tensor.unsqueeze(0).expand(B, -1, -1)

        # Encoder forward — grad flow controlled by requires_grad flags
        out_4d = self.reve(eeg, pos, return_output=False)

        # Rearrange to (B, C*H, E)
        x = rearrange(out_4d, "b c h e -> b (c h) e")

        if self.pooling == "last_avg":
            x = x.mean(dim=1)  # (B, E)
            return self.linear_head(x)

        elif self.pooling == "no":
            query = self.cls_query_token.expand(B, -1, -1)
            attn_scores = torch.matmul(query, x.transpose(-1, -2)) / (self.embed_dim**0.5)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            context = torch.matmul(attn_weights, x)

            x = torch.cat([context, x], dim=1)
            x = x.reshape(B, -1)
            return self.linear_head(x)

        else:  # "last"
            query = self.cls_query_token.expand(B, -1, -1)
            attn_scores = torch.matmul(query, x.transpose(-1, -2)) / (self.embed_dim**0.5)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            context = torch.matmul(attn_weights, x).squeeze(1)
            return self.linear_head(context)

    def freeze_encoder(self) -> None:
        """LP mode: freeze all encoder params, keep cls_query_token + head trainable."""
        for param in self.reve.parameters():
            param.requires_grad_(False)
        self.cls_query_token.requires_grad_(True)
        for param in self.linear_head.parameters():
            param.requires_grad_(True)

    def unfreeze_encoder(self) -> None:
        """FT mode: unfreeze all params (LoRA will be applied after this)."""
        for param in self.reve.parameters():
            param.requires_grad_(True)

    def set_dropout(self, dropout: float) -> None:
        """Update dropout rate in the linear head between stages."""
        for module in self.linear_head.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout

    def trainable_parameters(self):
        """Yield all parameters with requires_grad=True."""
        return (p for p in self.parameters() if p.requires_grad)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
