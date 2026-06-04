"""Tests for the statistical helpers in src.inference.statistical_tests.

Covers the pure-math functions that produce numbers in docs/statistical_tests*.md:
  - holm_bonferroni
  - bca_bootstrap_ci
  - bca_variance_ratio_ci
  - run_friedman
  - run_brown_forsythe_friedman

These tests are fast (<1 s each), deterministic, and use synthetic vectors.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from src.inference.statistical_tests import (
    FriedmanResult,
    bca_bootstrap_ci,
    bca_variance_ratio_ci,
    holm_bonferroni,
    run_brown_forsythe_friedman,
    run_friedman,
)

# ── holm_bonferroni ────────────────────────────────────────────────────────


class TestHolmBonferroni:
    def test_already_sorted_input(self):
        # Classical Holm example: p = [0.01, 0.02, 0.03, 0.04], m=4.
        # k-th smallest multiplied by (m-k+1):
        #   0.01 * 4 = 0.04
        #   0.02 * 3 = 0.06  -> stays 0.06 (monotone OK)
        #   0.03 * 2 = 0.06  -> max with prior → 0.06
        #   0.04 * 1 = 0.04  -> max with prior → 0.06
        adj = holm_bonferroni([0.01, 0.02, 0.03, 0.04])
        np.testing.assert_allclose(adj, [0.04, 0.06, 0.06, 0.06])

    def test_unsorted_input_preserves_order(self):
        # Same p-values but in different order — output must align with INPUT positions.
        adj = holm_bonferroni([0.04, 0.01, 0.03, 0.02])
        np.testing.assert_allclose(adj, [0.06, 0.04, 0.06, 0.06])

    def test_cap_at_1(self):
        # Large p-values must be capped at 1.0.
        adj = holm_bonferroni([0.6, 0.5, 0.4])
        assert all(p <= 1.0 for p in adj)
        # 0.4 * 3 = 1.2 → 1.0; subsequent values pinned via running_max.
        np.testing.assert_allclose(adj, [1.0, 1.0, 1.0])

    def test_single_p_unchanged(self):
        # m=1: multiplier is 1; nothing changes.
        assert holm_bonferroni([0.03]) == [0.03]

    def test_monotonicity_property(self):
        # The Holm-adjusted p-values must be monotone non-decreasing in the SORT
        # order of the input. Verify against random inputs.
        rng = np.random.default_rng(0)
        for _ in range(20):
            p = rng.uniform(0, 1, size=rng.integers(2, 20)).tolist()
            adj = holm_bonferroni(p)
            order = np.argsort(p)
            sorted_adj = [adj[i] for i in order]
            assert all(
                sorted_adj[i] <= sorted_adj[i + 1] + 1e-12 for i in range(len(sorted_adj) - 1)
            )

    def test_extremely_small_pvalues(self):
        adj = holm_bonferroni([1e-15, 0.05])
        np.testing.assert_allclose(adj, [2e-15, 0.05])


# ── bca_bootstrap_ci ───────────────────────────────────────────────────────


class TestBcaBootstrapCi:
    def test_returns_two_floats_with_lo_lt_hi(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0.6, 0.1, size=100)
        lo, hi = bca_bootstrap_ci(x, seed=42, n_boot=500)
        assert isinstance(lo, float)
        assert isinstance(hi, float)
        assert lo < hi

    def test_ci_brackets_observed_mean(self):
        # For a roughly normal sample, the CI should contain the sample mean.
        rng = np.random.default_rng(0)
        x = rng.normal(0.6, 0.1, size=100)
        lo, hi = bca_bootstrap_ci(x, seed=42, n_boot=500)
        assert lo <= x.mean() <= hi

    def test_reproducibility_same_seed(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0.6, 0.1, size=100)
        a = bca_bootstrap_ci(x, seed=42, n_boot=500)
        b = bca_bootstrap_ci(x, seed=42, n_boot=500)
        assert a == b

    def test_different_seeds_give_different_results(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0.6, 0.1, size=100)
        a = bca_bootstrap_ci(x, seed=42, n_boot=500)
        b = bca_bootstrap_ci(x, seed=123, n_boot=500)
        # Different seeds → at least one endpoint differs.
        assert a != b

    def test_matches_scipy_directly(self):
        # Sanity: our wrapper should produce the same number as a direct
        # scipy.stats.bootstrap call with the same seed.
        rng_seed = 42
        x = np.array([0.55, 0.62, 0.71, 0.48, 0.66, 0.59, 0.73, 0.51, 0.69, 0.64])
        ours = bca_bootstrap_ci(x, seed=rng_seed, n_boot=1000)
        ref = stats.bootstrap(
            (x,),
            np.mean,
            n_resamples=1000,
            confidence_level=0.95,
            method="BCa",
            random_state=np.random.default_rng(rng_seed),
            vectorized=False,
        )
        np.testing.assert_allclose(ours[0], float(ref.confidence_interval.low))
        np.testing.assert_allclose(ours[1], float(ref.confidence_interval.high))

    def test_narrow_ci_for_large_n(self):
        # CI width shrinks as ~1/sqrt(n) — bigger n → tighter interval.
        rng = np.random.default_rng(0)
        x_small = rng.normal(0.6, 0.1, size=30)
        x_large = rng.normal(0.6, 0.1, size=300)
        lo_s, hi_s = bca_bootstrap_ci(x_small, seed=42, n_boot=500)
        lo_l, hi_l = bca_bootstrap_ci(x_large, seed=42, n_boot=500)
        assert (hi_l - lo_l) < (hi_s - lo_s)


# ── bca_variance_ratio_ci ──────────────────────────────────────────────────


class TestBcaVarianceRatioCi:
    def test_returns_three_floats_with_lo_lt_hi(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, size=100)
        b = rng.normal(0, 1, size=100)
        ratio, lo, hi = bca_variance_ratio_ci(a, b, seed=42, n_boot=500)
        assert isinstance(ratio, float)
        assert lo < hi

    def test_ratio_brackets_observed_value(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, size=100)
        b = rng.normal(0, 1, size=100)
        ratio, lo, hi = bca_variance_ratio_ci(a, b, seed=42, n_boot=500)
        # The point estimate should fall inside the CI.
        assert lo <= ratio <= hi

    def test_equal_variances_give_ratio_near_one(self):
        # If a and b have the same variance, the ratio CI should bracket 1.0.
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, size=200)
        b = rng.normal(0, 1, size=200)
        ratio, lo, hi = bca_variance_ratio_ci(a, b, seed=42, n_boot=1000)
        assert lo < 1.0 < hi

    def test_a_more_variable_gives_ratio_gt_one(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 3.0, size=200)  # larger spread
        b = rng.normal(0, 1.0, size=200)
        ratio, _, _ = bca_variance_ratio_ci(a, b, seed=42, n_boot=500)
        assert ratio > 1.0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            bca_variance_ratio_ci(np.zeros(10), np.zeros(11), seed=42, n_boot=100)

    def test_reproducible_with_seed(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, size=100)
        b = rng.normal(0, 1, size=100)
        r1 = bca_variance_ratio_ci(a, b, seed=42, n_boot=500)
        r2 = bca_variance_ratio_ci(a, b, seed=42, n_boot=500)
        assert r1 == r2


# ── run_friedman ──────────────────────────────────────────────────────────


class TestRunFriedman:
    def test_basic_shape(self):
        rng = np.random.default_rng(0)
        v = {
            "A": rng.normal(0.5, 0.1, size=30),
            "B": rng.normal(0.5, 0.1, size=30),
            "C": rng.normal(0.5, 0.1, size=30),
        }
        res = run_friedman(v)
        assert isinstance(res, FriedmanResult)
        assert res.k == 3
        assert res.n == 30
        assert res.df == 2
        assert 0.0 <= res.kendall_w <= 1.0
        assert set(res.mean_ranks.keys()) == {"A", "B", "C"}

    def test_matches_scipy(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0.5, 0.1, size=20)
        b = rng.normal(0.5, 0.1, size=20)
        c = rng.normal(0.5, 0.1, size=20)
        res = run_friedman({"A": a, "B": b, "C": c})
        ref = stats.friedmanchisquare(a, b, c)
        np.testing.assert_allclose(res.stat, ref.statistic)
        np.testing.assert_allclose(res.pval, ref.pvalue)

    def test_strong_difference_rejects_h0(self):
        # If one method dominates on every subject, Friedman should reject H0.
        n = 50
        a = np.full(n, 0.40)
        b = np.full(n, 0.60)
        c = np.full(n, 0.80)
        # Add tiny noise so ranks aren't degenerate
        rng = np.random.default_rng(0)
        a += rng.normal(0, 0.001, size=n)
        b += rng.normal(0, 0.001, size=n)
        c += rng.normal(0, 0.001, size=n)
        res = run_friedman({"A": a, "B": b, "C": c})
        assert res.pval < 1e-10
        # Mean ranks should reflect the ordering A<B<C → ranks 1,2,3.
        assert res.mean_ranks["A"] < res.mean_ranks["B"] < res.mean_ranks["C"]

    def test_unequal_lengths_raise(self):
        with pytest.raises(ValueError):
            run_friedman({"A": np.zeros(10), "B": np.zeros(11), "C": np.zeros(10)})

    def test_k_less_than_3_raises(self):
        with pytest.raises(ValueError):
            run_friedman({"A": np.zeros(10), "B": np.zeros(10)})

    def test_kendall_w_formula(self):
        # W = chi2 / (n * (k-1)).
        rng = np.random.default_rng(0)
        v = {f"M{i}": rng.normal(0.5, 0.1, size=40) for i in range(3)}
        res = run_friedman(v)
        expected_w = res.stat / (res.n * (res.k - 1))
        np.testing.assert_allclose(res.kendall_w, expected_w)


# ── run_brown_forsythe_friedman ───────────────────────────────────────────


class TestRunBrownForsytheFriedman:
    def test_equal_dispersion_does_not_reject(self):
        # All three vectors have the same variance → BF-Friedman should not reject.
        rng = np.random.default_rng(0)
        n = 200
        v = {
            "A": rng.normal(0.5, 0.1, size=n),
            "B": rng.normal(0.6, 0.1, size=n),  # different mean, same std
            "C": rng.normal(0.4, 0.1, size=n),
        }
        res = run_brown_forsythe_friedman(v)
        # Not asserting > 0.05 strictly (random fluctuation) but should be far from tiny
        assert res.pval > 0.01

    def test_different_dispersion_rejects(self):
        rng = np.random.default_rng(0)
        n = 200
        v = {
            "A": rng.normal(0.5, 0.05, size=n),  # tight
            "B": rng.normal(0.5, 0.30, size=n),  # very wide
            "C": rng.normal(0.5, 0.05, size=n),
        }
        res = run_brown_forsythe_friedman(v)
        assert res.pval < 0.05

    def test_returns_friedman_result_type(self):
        rng = np.random.default_rng(0)
        v = {f"M{i}": rng.normal(0.5, 0.1, size=30) for i in range(3)}
        res = run_brown_forsythe_friedman(v)
        assert isinstance(res, FriedmanResult)
        assert res.k == 3

    def test_does_not_mutate_input(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0.5, 0.1, size=30)
        a_copy = a.copy()
        run_brown_forsythe_friedman({"A": a, "B": a.copy(), "C": a.copy()})
        np.testing.assert_array_equal(a, a_copy)
