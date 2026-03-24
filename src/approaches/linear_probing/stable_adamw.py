"""
stable_adamw.py — Simplified StableAdamW optimizer for linear probing.

Stripped of DDP, gradient_release, and norm-return features.
Keeps the core algorithm: debiased betas, RMS-stabilized learning rates,
decoupled weight decay.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


def _debias_beta(beta: float, step: int) -> float:
    """Adam-style debiased beta: betahat = beta*(1-beta**(step-1))/(1-beta**step)."""
    return (beta**step - beta) / (beta**step - 1)


class StableAdamW(Optimizer):
    """StableAdamW: AdamW-Adafactor hybrid with RMS-stabilized learning rates.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 5e-3)
        betas: Coefficients for gradient and squared gradient moving averages
               (default: (0.92, 0.999))
        weight_decay: Decoupled weight decay coefficient (default: 0.01)
        eps: Numerical stability term (default: 1e-9)
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict],
        lr: float = 5e-3,
        betas: tuple[float, float] = (0.92, 0.999),
        weight_decay: float = 0.01,
        eps: float = 1e-9,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps}")

        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1],
                        weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            lr = group["lr"]
            wd = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Lazy state init
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Debiased beta computation
                beta1_comp = 1 - _debias_beta(beta1, step)
                beta2_hat = _debias_beta(beta2, step)

                # Update moving averages
                exp_avg.lerp_(grad, weight=beta1_comp)
                exp_avg_sq.mul_(beta2_hat).addcmul_(grad, grad, value=1 - beta2_hat)

                # RMS stabilization: compute per-parameter RMS of update ratio
                eps_sq = eps ** 2
                denom_sq = torch.maximum(exp_avg_sq, torch.full_like(exp_avg_sq, eps_sq))
                rms = (grad.pow(2) / denom_sq).mean().sqrt().item()
                neg_lr = -lr / max(1.0, rms)

                # Decoupled weight decay
                if wd != 0:
                    p.mul_(1 + neg_lr * wd)

                # Adam update step
                denom = exp_avg_sq.sqrt().add_(eps)
                p.addcdiv_(exp_avg, denom, value=neg_lr)

        return loss
