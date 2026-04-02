"""
model.py — REVE classifier for JADE (Joint Alignment and Discriminative Embedding).

Provides ReveClassifierJADE: same architecture as ReveClassifierFT but with an
additional projection head for supervised contrastive learning.  The projection
head maps a compact representation (512-d or 1024-d) to a low-dimensional
L2-normalised embedding space used by SupConLoss.

The classifier head (RMSNorm → Dropout → Linear) produces logits as before;
the projection head (Linear → ReLU → Linear → L2-norm) produces embeddings
for contrastive learning.  Both heads share the same backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from src.approaches.shared.model_utils import RMSNorm, compute_n_patches


class ReveClassifierJADE(nn.Module):
    """
    Two-stage classifier with projection head for JADE joint training.

    Stage 1 (LP warmup): frozen encoder, CE only, projection head frozen.
    Stage 2 (FT):        LoRA/full FT + joint CE + SupCon loss.
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
        proj_dim: int = 128,
        proj_hidden: int = 512,
        supcon_repr: str = "context",
    ) -> None:
        super().__init__()

        assert pooling in ("no", "last", "last_avg"), f"Pooling '{pooling}' not supported"
        assert supcon_repr in ("context", "mean", "both"), (
            f"supcon_repr '{supcon_repr}' not supported"
        )

        self.pooling = pooling
        self.reve = reve_model
        self.embed_dim = reve_model.config.embed_dim  # 512 (Base)
        self.supcon_repr = supcon_repr

        # Register electrode positions as buffer
        self.register_buffer("pos_tensor", pos_tensor)  # (n_channels, 3)

        # Trainable query token — initialised from pretrained checkpoint
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

        # ── Projection head for SupCon ────────────────────────────────
        proj_input_dim = self.embed_dim if supcon_repr != "both" else self.embed_dim * 2
        self.projection_head = nn.Sequential(
            nn.Linear(proj_input_dim, proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_dim),
        )

    # ── Forward ───────────────────────────────────────────────────────

    def forward(
        self,
        eeg: Tensor,
        pos: Tensor | None = None,
        return_projections: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Args:
            eeg:  (B, C, T) raw EEG.
            pos:  (B, C, 3) electrode positions.  If None, uses registered buffer.
            return_projections: If True, also return L2-normalised projection
                                embeddings for the SupCon loss.

        Returns:
            logits (B, n_classes)                          — when return_projections=False
            (logits (B, n_classes), z (B, proj_dim))       — when return_projections=True
        """
        B = eeg.shape[0]

        if pos is None:
            pos = self.pos_tensor.unsqueeze(0).expand(B, -1, -1)

        # Encoder forward — grad flow controlled by requires_grad flags
        out_4d = self.reve(eeg, pos, return_output=False)

        # Rearrange to (B, C*H, E)
        x = rearrange(out_4d, "b c h e -> b (c h) e")

        # ── Pooling & classification ──────────────────────────────────

        if self.pooling == "last_avg":
            pool_repr = x.mean(dim=1)  # (B, E)
            logits = self.linear_head(pool_repr)

            if return_projections:
                proj_input = self._get_proj_input(x, context=None, mean_repr=pool_repr)
                z = F.normalize(self.projection_head(proj_input), dim=-1)
                return logits, z
            return logits

        # For "no" and "last": compute query attention
        query = self.cls_query_token.expand(B, -1, -1)
        attn_scores = torch.matmul(query, x.transpose(-1, -2)) / (self.embed_dim**0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, x)  # (B, 1, E)

        if self.pooling == "no":
            cls_input = torch.cat([context, x], dim=1).reshape(B, -1)
            logits = self.linear_head(cls_input)
        else:  # "last"
            logits = self.linear_head(context.squeeze(1))

        if return_projections:
            proj_input = self._get_proj_input(x, context=context.squeeze(1))
            z = F.normalize(self.projection_head(proj_input), dim=-1)
            return logits, z

        return logits

    # ── Projection representation selection ───────────────────────────

    def _get_proj_input(
        self,
        x: Tensor,
        context: Tensor | None = None,
        mean_repr: Tensor | None = None,
    ) -> Tensor:
        """Select the compact representation for the projection head.

        Args:
            x:         (B, C*H, E)  all patch tokens.
            context:   (B, E)       query-attention context (None for last_avg).
            mean_repr: (B, E)       mean-pooled patches (None if not yet computed).
        """
        if mean_repr is None:
            mean_repr = x.mean(dim=1)

        if self.supcon_repr == "mean":
            return mean_repr

        # Compute context on-the-fly when not already available (last_avg path)
        if context is None:
            B = x.shape[0]
            query = self.cls_query_token.expand(B, -1, -1)
            attn = torch.softmax(
                torch.matmul(query, x.transpose(-1, -2)) / (self.embed_dim**0.5),
                dim=-1,
            )
            context = torch.matmul(attn, x).squeeze(1)  # (B, E)

        if self.supcon_repr == "context":
            return context
        else:  # "both"
            return torch.cat([context, mean_repr], dim=-1)  # (B, 2*E)

    # ── Freeze / unfreeze helpers ─────────────────────────────────────

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

    def freeze_projection_head(self) -> None:
        """Freeze projection head (LP warmup: SupCon not active)."""
        for param in self.projection_head.parameters():
            param.requires_grad_(False)

    def unfreeze_projection_head(self) -> None:
        """Unfreeze projection head (FT stage: SupCon active)."""
        for param in self.projection_head.parameters():
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
