"""Tests for run_paired in src.inference.statistical_tests.

run_paired is the per-comparison driver behind §2 of docs/statistical_tests.md:
it computes the paired t-test, paired Wilcoxon, sign test, both CI flavours,
Cohen's d, rank-biserial r, and the distributional diagnostics in one shot.

Failures here directly corrupt thesis significance claims.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from src.inference.statistical_tests import PairedResult, fmt_p, run_paired

# Reusable synthetic vectors

RNG = np.random.default_rng(0)


def _make_paired(n=100, delta=0.05, noise=0.1, ties=0, seed=0):
    """a = b + delta + noise, with `ties` exact-zero positions inserted."""
    rng = np.random.default_rng(seed)
    b = rng.normal(0.6, 0.1, size=n)
    a = b + delta + rng.normal(0, noise, size=n)
    if ties > 0:
        # Force `ties` positions to have a_i == b_i.
        idx = rng.choice(n, size=ties, replace=False)
        a[idx] = b[idx]
    return a, b


class TestRunPairedShape:
    def test_returns_paired_result(self):
        a, b = _make_paired()
        r = run_paired("A", a, "B", b, "test_task", seed=42)
        assert isinstance(r, PairedResult)
        assert r.label_a == "A"
        assert r.label_b == "B"
        assert r.task == "test_task"
        assert r.n == len(a)

    def test_means_match(self):
        a, b = _make_paired()
        r = run_paired("A", a, "B", b, "t", seed=42)
        np.testing.assert_allclose(r.mean_a, a.mean())
        np.testing.assert_allclose(r.mean_b, b.mean())
        np.testing.assert_allclose(r.mean_diff, (a - b).mean())

    def test_median_diff_matches(self):
        a, b = _make_paired()
        r = run_paired("A", a, "B", b, "t", seed=42)
        np.testing.assert_allclose(r.median_diff, float(np.median(a - b)))

    def test_std_diff_uses_ddof_1(self):
        a, b = _make_paired()
        r = run_paired("A", a, "B", b, "t", seed=42)
        np.testing.assert_allclose(r.std_diff, float((a - b).std(ddof=1)))


class TestWinsLossesTies:
    def test_counts_sum_to_n(self):
        a, b = _make_paired(ties=5)
        r = run_paired("A", a, "B", b, "t", seed=42)
        assert r.n_wins + r.n_losses + r.n_ties == r.n

    def test_ties_detected(self):
        a, b = _make_paired(ties=7)
        r = run_paired("A", a, "B", b, "t", seed=42)
        assert r.n_ties == 7
        assert r.n_nonzero == r.n - r.n_ties

    def test_wins_when_a_dominates(self):
        # Construct so a > b on every paired sample.
        n = 50
        b = np.linspace(0.5, 0.7, n)
        a = b + 0.05
        r = run_paired("A", a, "B", b, "t", seed=42)
        assert r.n_wins == n
        assert r.n_losses == 0
        assert r.n_ties == 0


class TestParametricBlock:
    def test_t_matches_scipy(self):
        a, b = _make_paired()
        r = run_paired("A", a, "B", b, "t", seed=42)
        ref = stats.ttest_rel(a, b)
        np.testing.assert_allclose(r.t_stat, float(ref.statistic))
        np.testing.assert_allclose(r.t_pval, float(ref.pvalue))

    def test_t_ci_brackets_mean_diff_when_significant(self):
        a, b = _make_paired(delta=0.5, noise=0.05, n=80)
        r = run_paired("A", a, "B", b, "t", seed=42)
        assert r.t_ci_lo < r.mean_diff < r.t_ci_hi

    def test_t_ci_formula(self):
        a, b = _make_paired(n=60)
        r = run_paired("A", a, "B", b, "t", seed=42)
        # Reconstruct: CI = mean_diff ± t_crit * std_diff / sqrt(n)
        se = r.std_diff / math.sqrt(r.n)
        t_crit = float(stats.t.ppf(0.975, df=r.n - 1))
        np.testing.assert_allclose(r.t_ci_hi - r.mean_diff, t_crit * se, atol=1e-10)
        np.testing.assert_allclose(r.mean_diff - r.t_ci_lo, t_crit * se, atol=1e-10)


class TestWilcoxonBlock:
    def test_matches_scipy(self):
        a, b = _make_paired()
        r = run_paired("A", a, "B", b, "t", seed=42)
        ref = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        np.testing.assert_allclose(r.w_stat, float(ref.statistic))
        np.testing.assert_allclose(r.w_pval, float(ref.pvalue))

    def test_rejects_strong_signal(self):
        a, b = _make_paired(delta=0.3, noise=0.05, n=80)
        r = run_paired("A", a, "B", b, "t", seed=42)
        assert r.w_pval < 1e-5


class TestSignTest:
    def test_nan_when_all_tied(self):
        a = np.array([0.5, 0.6, 0.7, 0.8])
        b = a.copy()
        r = run_paired("A", a, "B", b, "t", seed=42)
        assert math.isnan(r.sign_pval)
        assert r.n_nonzero == 0

    def test_extreme_dominance_gives_tiny_p(self):
        n = 60
        b = np.linspace(0.5, 0.7, n)
        a = b + 0.05  # a > b on every sample
        r = run_paired("A", a, "B", b, "t", seed=42)
        assert r.sign_pval < 1e-10

    def test_p_matches_binomtest(self):
        a, b = _make_paired(delta=0.05, noise=0.1, ties=3)
        r = run_paired("A", a, "B", b, "t", seed=42)
        ref = stats.binomtest(r.n_wins, r.n_nonzero, p=0.5, alternative="two-sided")
        np.testing.assert_allclose(r.sign_pval, float(ref.pvalue))


class TestEffectSizes:
    def test_cohen_d_formula(self):
        a, b = _make_paired()
        r = run_paired("A", a, "B", b, "t", seed=42)
        np.testing.assert_allclose(r.cohen_d, r.mean_diff / r.std_diff)

    def test_cohen_d_sign_matches_direction(self):
        # When a < b on average, Cohen's d should be negative.
        a = np.array([0.4, 0.5, 0.6])
        b = np.array([0.7, 0.8, 0.9])
        r = run_paired("A", a, "B", b, "t", seed=42)
        assert r.cohen_d < 0

    def test_rank_biserial_bounds(self):
        a, b = _make_paired()
        r = run_paired("A", a, "B", b, "t", seed=42)
        assert -1.0 <= r.rank_biserial <= 1.0

    def test_rank_biserial_extreme_dominance(self):
        n = 40
        b = np.linspace(0.5, 0.7, n)
        a = b + 0.05  # a always beats b → rb should be +1.
        r = run_paired("A", a, "B", b, "t", seed=42)
        np.testing.assert_allclose(r.rank_biserial, 1.0)

    def test_rank_biserial_nan_when_all_tied(self):
        a = np.array([0.5, 0.6, 0.7])
        b = a.copy()
        r = run_paired("A", a, "B", b, "t", seed=42)
        assert math.isnan(r.rank_biserial)


class TestDiagnostics:
    def test_skew_matches_scipy(self):
        a, b = _make_paired()
        r = run_paired("A", a, "B", b, "t", seed=42)
        np.testing.assert_allclose(r.diff_skew, float(stats.skew(a - b)))

    def test_kurtosis_is_excess(self):
        a, b = _make_paired()
        r = run_paired("A", a, "B", b, "t", seed=42)
        # scipy.stats.kurtosis defaults to "excess" → normal distribution gives ~0.
        np.testing.assert_allclose(r.diff_kurtosis_excess, float(stats.kurtosis(a - b)))

    def test_shapiro_matches_scipy(self):
        a, b = _make_paired()
        r = run_paired("A", a, "B", b, "t", seed=42)
        np.testing.assert_allclose(r.shapiro_pval, float(stats.shapiro(a - b).pvalue))


class TestReproducibility:
    def test_same_seed_same_bootstrap_ci(self):
        a, b = _make_paired()
        r1 = run_paired("A", a, "B", b, "t", seed=42)
        r2 = run_paired("A", a, "B", b, "t", seed=42)
        assert r1.boot_ci_lo == r2.boot_ci_lo
        assert r1.boot_ci_hi == r2.boot_ci_hi

    def test_different_seed_different_bootstrap(self):
        a, b = _make_paired()
        r1 = run_paired("A", a, "B", b, "t", seed=42)
        r2 = run_paired("A", a, "B", b, "t", seed=123)
        assert (r1.boot_ci_lo, r1.boot_ci_hi) != (r2.boot_ci_lo, r2.boot_ci_hi)


class TestFmtP:
    def test_nan_renders_dash(self):
        assert fmt_p(float("nan")) == "—"

    def test_below_threshold_uses_scientific(self):
        assert fmt_p(1e-5) == "1.00e-05"

    def test_small_uses_lt_marker(self):
        assert fmt_p(5e-4) == "<0.001"

    def test_normal_uses_3_decimal(self):
        assert fmt_p(0.05) == "0.050"

    @pytest.mark.parametrize("p", [0.5, 0.99, 0.001, 1e-3])
    def test_returns_string(self, p):
        assert isinstance(fmt_p(p), str)
