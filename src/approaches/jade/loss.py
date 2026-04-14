"""
loss.py — Supervised Contrastive Loss.

Implements L_sup_out: for each anchor i, averages the log-ratio over all
positive pairs P(i), with the denominator summing over all non-self pairs A(i).

Reference: "Supervised Contrastive Learning", Khosla et al. 2020, Eq. 2.
https://doi.org/10.48550/arXiv.2004.11362
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.

    For a batch of B samples with L2-normalised projections z and labels y::

        L = (1/B') sum_{i where |P(i)|>0}
              -1/|P(i)| sum_{p in P(i)}
                log( exp(z_i . z_p / tau) / sum_{a in A(i)} exp(z_i . z_a / tau) )

    where P(i) = {j : j!=i, y_j==y_i}  (positives)
          A(i) = {j : j!=i}             (all except self)
          B'   = number of anchors with at least one positive
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        """
        Args:
            features: (B, D) L2-normalised projection embeddings.
            labels:   (B,)   integer class labels.

        Returns:
            Scalar loss averaged over anchors that have at least one positive.
        """
        # Cast to float32 for numerical safety (exp/log can overflow in fp16)
        features = features.float()
        device = features.device
        B = features.shape[0]

        # Pairwise cosine similarity (features already L2-normed)
        sim = features @ features.T / self.temperature  # (B, B)

        # Masks
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        self_mask = ~torch.eye(B, dtype=torch.bool, device=device)  # A(i) (all except self)
        pos_mask = labels_eq & self_mask  # P(i) (same class, not self)

        # Log-sum-exp trick for numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - sim_max.detach()  # (B, B)

        # Denominator: sum_{a in A(i)} exp(z_i . z_a / tau)
        exp_logits = torch.exp(logits) * self_mask.float()
        log_sum_exp = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Log-probability for every pair
        log_prob = logits - log_sum_exp  # (B, B)

        # Average log-prob over positive pairs per anchor
        n_positives = pos_mask.float().sum(dim=1)  # (B,)
        valid = n_positives > 0  # exclude singleton-class anchors

        mean_log_prob = (pos_mask.float() * log_prob).sum(dim=1) / n_positives.clamp(min=1)

        # If no anchor has a positive pair (all singletons), return zero loss.
        if not valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = -(mean_log_prob[valid]).mean()
        return loss
