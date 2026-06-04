"""Tests for SupConLoss (Khosla 2020, L_sup_out).

SupConLoss lives at src.approaches.jade.loss.SupConLoss and is the actual
training signal that drives JADE's contrastive term. A bug here silently
changes what the encoder learns and which configurations win the HP sweep.

Tests cover:
  - Output shape + scalar guarantee
  - Numerical reference value against a hand-computable case
  - Symmetry / invariance properties (label permutation, batch order)
  - Singleton-class handling (no positives → zero gradient contribution)
  - Temperature scaling behaviour
  - Differentiability (gradient flows into the features)
  - Numerical stability under small tau and large batches
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.approaches.jade.loss import SupConLoss


def _normed_features(batch: int, dim: int, seed: int = 0) -> torch.Tensor:
    """Random L2-normalised features (the input the projection head produces)."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(batch, dim, generator=g)
    return F.normalize(z, dim=-1)


# ── Shape and scalar guarantees ───────────────────────────────────────────


class TestShape:
    def test_returns_scalar_tensor(self):
        loss_fn = SupConLoss(temperature=0.1)
        z = _normed_features(16, 64)
        y = torch.tensor([0, 1, 0, 1, 2, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        out = loss_fn(z, y)
        assert isinstance(out, torch.Tensor)
        assert out.shape == ()  # scalar

    def test_loss_is_non_negative(self):
        # SupCon loss is a negative log-prob average → must be ≥ 0.
        loss_fn = SupConLoss(temperature=0.1)
        z = _normed_features(32, 64)
        y = torch.arange(32) % 4  # 4 classes
        out = loss_fn(z, y)
        assert out.item() >= 0.0

    def test_loss_is_finite(self):
        loss_fn = SupConLoss(temperature=0.1)
        z = _normed_features(32, 64)
        y = torch.arange(32) % 4
        out = loss_fn(z, y)
        assert math.isfinite(out.item())


# ── Singleton handling ────────────────────────────────────────────────────


class TestSingletons:
    def test_all_singleton_classes_returns_zero(self):
        # Each sample is its own class → no positives anywhere.
        loss_fn = SupConLoss(temperature=0.1)
        z = _normed_features(8, 32)
        y = torch.arange(8)  # unique labels
        out = loss_fn(z, y)
        assert out.item() == 0.0
        # Should still allow gradient — used for safety in the training loop.
        assert out.requires_grad

    def test_singleton_class_excluded_from_mean(self):
        # 8 samples: 7 share label 0, 1 has label 1 (singleton).
        # The singleton anchor must not contribute to the mean.
        loss_fn = SupConLoss(temperature=0.1)
        z = _normed_features(8, 32)
        y = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1])  # one singleton

        loss_singleton = loss_fn(z, y).item()

        # Same labels, but remove the singleton entirely.
        loss_no_singleton = loss_fn(z[:7], y[:7]).item()

        # The singleton's positives don't exist, so the only difference is the
        # extra negative pair in the denominators of the other 7 anchors. The
        # values should be close but not identical.
        assert abs(loss_singleton - loss_no_singleton) < 1.0  # sanity bound
        assert loss_singleton > 0.0


# ── Permutation / order invariance ────────────────────────────────────────


class TestInvariances:
    def test_label_permutation_invariance(self):
        # SupCon depends only on which samples share a class, not on the
        # specific integer used as the class label. Relabelling preserves loss.
        loss_fn = SupConLoss(temperature=0.1)
        z = _normed_features(16, 64)
        y = torch.tensor([0, 1, 0, 1, 2, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

        # Permute label values: 0→7, 1→3, 2→5.
        y_relabeled = torch.tensor([{0: 7, 1: 3, 2: 5}[int(v)] for v in y])
        out_orig = loss_fn(z, y).item()
        out_relabel = loss_fn(z, y_relabeled).item()
        assert abs(out_orig - out_relabel) < 1e-6

    def test_batch_order_invariance(self):
        # Shuffling rows of (z, y) together must give the same loss.
        loss_fn = SupConLoss(temperature=0.1)
        z = _normed_features(16, 64)
        y = torch.tensor([0, 1, 0, 1, 2, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

        perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(7))
        z_p = z[perm]
        y_p = y[perm]
        np_orig = loss_fn(z, y).item()
        np_perm = loss_fn(z_p, y_p).item()
        assert abs(np_orig - np_perm) < 1e-6


# ── Numerical reference (hand-computable case) ────────────────────────────


class TestNumericalReference:
    def test_two_classes_two_each(self):
        """Hand-computable reference: 4 samples, 2 classes (2 of each).

        Choose features so the cosine-similarity matrix is exact. With
        unit-norm features at 0°/180° splits, the math collapses cleanly.
        """
        # 4 samples in a 2D unit-norm hypersphere.
        z = torch.tensor(
            [
                [1.0, 0.0],  # class 0
                [1.0, 0.0],  # class 0 (identical → cos = 1)
                [0.0, 1.0],  # class 1
                [0.0, 1.0],  # class 1
            ]
        )
        y = torch.tensor([0, 0, 1, 1])
        tau = 0.5
        loss_fn = SupConLoss(temperature=tau)
        out = loss_fn(z, y).item()

        # Compute by hand for one anchor (e.g. index 0, class 0):
        #   positives P(0) = {1}
        #   A(0) = {1, 2, 3}
        #   sim(0,1) = 1.0 ;  sim(0,2) = 0.0 ;  sim(0,3) = 0.0
        #   logits / tau = (2.0, 0.0, 0.0)  before max-subtraction
        # After max-subtract (max=2.0): (0.0, -2.0, -2.0)
        # denominator = exp(0) + exp(-2) + exp(-2) = 1 + 2*exp(-2)
        # log-prob for the positive pair = 0 - log(1 + 2*exp(-2))
        # ell_0 = -[0 - log(1 + 2*exp(-2))] = log(1 + 2*exp(-2))
        # By symmetry every anchor has the same loss → mean = log(1 + 2*exp(-2)).
        expected = math.log(1 + 2 * math.exp(-2))
        assert abs(out - expected) < 1e-5

    def test_perfectly_clustered_features_floor_at_log_p(self):
        """When same-class features are identical and different-class features
        are anti-aligned, the loss converges to log(|P(i)|).

        With |P(i)| = 3 same-class samples (after excluding self), positives
        compete with each other in the softmax denominator, so the floor is
        log(3) rather than 0. This pins the irreducible-positive-competition
        behaviour of L_sup_out.
        """
        # Two well-separated clusters of 4 samples each.
        z = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [-1.0, 0.0],
                [-1.0, 0.0],
                [-1.0, 0.0],
                [-1.0, 0.0],
            ]
        )
        y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        out = SupConLoss(temperature=0.1)(z, y).item()
        # Each anchor has 3 positives, max-subtraction zeroes the positive logits
        # and the 4 negatives become exp(-20) ≈ 0 → denominator ≈ 3 → loss ≈ log(3).
        assert abs(out - math.log(3)) < 1e-3

    def test_random_features_higher_loss(self):
        # Random features → loss should be substantially > the clustered case.
        z = _normed_features(8, 64, seed=42)
        y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        out = SupConLoss(temperature=0.1)(z, y).item()
        assert out > 0.5


# ── Temperature behaviour ─────────────────────────────────────────────────


class TestTemperature:
    def test_temperature_changes_loss(self):
        # Different temperatures should produce different loss values.
        z = _normed_features(16, 64)
        y = torch.arange(16) % 4
        l_low = SupConLoss(temperature=0.05)(z, y).item()
        l_high = SupConLoss(temperature=0.5)(z, y).item()
        assert l_low != l_high

    def test_constructor_stores_temperature(self):
        for t in (0.01, 0.07, 0.1, 0.5, 1.0):
            loss_fn = SupConLoss(temperature=t)
            assert loss_fn.temperature == t

    def test_default_temperature(self):
        # Default is set to 0.1 in the JADE pipeline.
        assert SupConLoss().temperature == 0.1


# ── Gradient flow ─────────────────────────────────────────────────────────


class TestGradientFlow:
    def test_gradient_flows_to_features(self):
        z = _normed_features(16, 64).requires_grad_(True)
        y = torch.arange(16) % 4
        loss = SupConLoss(temperature=0.1)(z, y)
        loss.backward()
        assert z.grad is not None
        assert z.grad.shape == z.shape
        # Some entries must be non-zero (anchors with positives).
        assert z.grad.abs().sum() > 0.0

    def test_singleton_only_loss_is_safe_to_backward(self):
        # When every class is a singleton, the loss returns a fresh
        # `torch.tensor(0.0, requires_grad=True)` that is NOT in the
        # autograd graph of z — so z.grad stays None. The important contract
        # is that .backward() doesn't crash, so the training loop is safe.
        z = _normed_features(8, 64).requires_grad_(True)
        y = torch.arange(8)  # all singletons
        loss = SupConLoss(temperature=0.1)(z, y)
        loss.backward()  # must not raise
        # z.grad is None because no path from loss back to z exists.
        assert z.grad is None
        assert loss.item() == 0.0


# ── Numerical stability ──────────────────────────────────────────────────


class TestNumericalStability:
    @pytest.mark.parametrize("tau", [0.01, 0.05, 0.1, 0.5, 1.0])
    def test_finite_across_temperatures(self, tau):
        z = _normed_features(32, 128, seed=11)
        y = torch.arange(32) % 4
        out = SupConLoss(temperature=tau)(z, y).item()
        assert math.isfinite(out)

    def test_finite_with_large_batch(self):
        z = _normed_features(256, 128, seed=11)
        y = torch.arange(256) % 9  # 9 classes
        out = SupConLoss(temperature=0.07)(z, y).item()
        assert math.isfinite(out)

    def test_does_not_overflow_with_small_tau(self):
        # Small tau pushes exp arguments to large positive/negative values.
        # The log-sum-exp trick in the loss should prevent overflow.
        z = _normed_features(64, 128, seed=11)
        y = torch.arange(64) % 4
        out = SupConLoss(temperature=0.01)(z, y).item()
        assert math.isfinite(out)
        assert out >= 0.0
