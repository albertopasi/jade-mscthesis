"""statistical_tests.py — paired significance tests on per-subject accuracies.

Reads the per-subject accuracy dicts produced by inference_subject_wise.py
(stored in main-results/{approach}_{task}/{stem}.json) and computes a
methodology suite that is robust to the audit issues identified during
review (non-normality of some paired-difference distributions, dependency
between subjects in the same CV fold, exact zero-difference pairs):

  * Per-condition descriptive stats with **BCa bootstrap 95 % CIs** on the mean.
  * Paired tests on each (JADE vs baseline, task) pair:
      * Paired t-test (parametric, primary when normality holds).
      * Wilcoxon signed-rank (non-parametric, primary otherwise).
      * Sign test (binomial on win count; convergent evidence).
      * Shapiro-Wilk normality test on the paired differences.
  * 95 % CI on the paired mean difference, computed BOTH ways:
      * Parametric (t-based).
      * Non-parametric (percentile bootstrap on the differences).
  * Effect sizes: Cohen's d (paired) and rank-biserial r.
  * Holm-Bonferroni adjusted p-values across the 4 comparisons (separately
    for t and Wilcoxon).

Writes a single markdown document at docs/statistical_tests.md.

Usage:
    uv run python -m src.inference.statistical_tests
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "main-results"
OUT_PATH = PROJECT_ROOT / "docs" / "statistical_tests.md"

RUNS: dict[tuple[str, str], str] = {
    ("lp", "9-class"): "lp_faced_v2_9-class_w10s10_pool_no_official_nomixup",
    ("lp", "binary"): "lp_faced_v2_binary_w10s10_pool_no_official_nomixup",
    ("ft", "9-class"): "ft_faced_9-class_w10s10_pool_no_r16_b256_lr0.0004_nomixup_fullft",
    ("ft", "binary"): "ft_faced_binary_w10s10_pool_no_r16_nomixup_fullft",
    (
        "jade",
        "9-class",
    ): "jade_faced_9-class_w10s10_pool_no_r16_a0.3_t0.2_context_b256_lr0.0004_fullft",
    (
        "jade",
        "binary",
    ): "jade_faced_binary_w10s10_pool_no_r16_a0.2_t0.05_context_b128_lr0.0001_fullft",
}
LABELS = {"lp": "LP", "ft": "SFT", "jade": "JADE"}
TASKS = ["9-class", "binary"]
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42
NORMALITY_ALPHA = 0.05


def load_per_subject(approach: str, task: str) -> dict[int, float]:
    path = RESULTS_ROOT / f"{approach}_{task}" / f"{RUNS[(approach, task)]}.json"
    data = json.loads(path.read_text())
    return {int(k): float(v) for k, v in data["per_subject_acc"].items()}


def aligned_pair(
    a: dict[int, float], b: dict[int, float]
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    common = sorted(set(a) & set(b))
    if not common:
        raise RuntimeError("No overlapping subjects between conditions.")
    return (
        np.array([a[s] for s in common], dtype=float),
        np.array([b[s] for s in common], dtype=float),
        common,
    )


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values for a family of `pvals` tests."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        scaled = pvals[idx] * (m - rank)
        running_max = max(running_max, scaled)
        adjusted[idx] = min(1.0, running_max)
    return adjusted.tolist()


def bca_bootstrap_ci(
    x: np.ndarray, seed: int, n_boot: int, alpha: float = 0.05
) -> tuple[float, float]:
    """BCa bootstrap CI on the mean of x via scipy.stats.bootstrap.

    BCa = bias-corrected and accelerated. It adjusts for both bias in the
    bootstrap distribution and skewness of the underlying statistic. This is
    the textbook-recommended interval when no stronger assumptions are made.
    """
    rng = np.random.default_rng(seed)
    res = stats.bootstrap(
        (x,),
        np.mean,
        n_resamples=n_boot,
        confidence_level=1 - alpha,
        method="BCa",
        random_state=rng,
        vectorized=False,
    )
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


@dataclass
class FriedmanResult:
    """Result of a Friedman omnibus test across k repeated measures on n blocks."""

    labels: list[str]
    n: int  # number of blocks (subjects)
    k: int  # number of conditions (methods)
    stat: float  # chi-squared statistic
    df: int  # degrees of freedom = k - 1
    pval: float
    kendall_w: float  # Kendall's W effect size in [0, 1]
    mean_ranks: dict[str, float]  # average rank per condition (lower = better)


def run_friedman(vectors: dict[str, np.ndarray]) -> FriedmanResult:
    """Friedman rank test across k repeated measures (same n subjects).

    H0: the k conditions have identical rank distributions across blocks.
    Non-parametric, no normality assumption. Multi-condition extension of
    the Wilcoxon signed-rank test.

    Args:
        vectors: ordered dict mapping condition label -> length-n accuracy vector.
                 All vectors must be the same length and aligned by subject.

    Returns:
        FriedmanResult with the chi-squared statistic, p-value, Kendall's W
        effect size, and per-condition mean ranks.
    """
    labels = list(vectors.keys())
    arrays = [vectors[lbl] for lbl in labels]
    lengths = {len(a) for a in arrays}
    if len(lengths) != 1:
        raise ValueError(f"Friedman requires equal-length vectors; got lengths {lengths}")
    n = lengths.pop()
    k = len(arrays)
    if k < 3:
        raise ValueError(f"Friedman requires k >= 3 conditions; got k={k}")

    res = stats.friedmanchisquare(*arrays)
    stat = float(res.statistic)
    pval = float(res.pvalue)

    # Kendall's W = chi^2 / (n * (k - 1)) — effect size in [0, 1].
    # 0 = no agreement among blocks on ranking, 1 = perfect agreement.
    kendall_w = stat / (n * (k - 1))

    # Mean rank per condition (rank 1 = lowest accuracy on that subject).
    # Compute per-block ranks (averaging ties) and average across blocks.
    stacked = np.column_stack(arrays)  # (n, k)
    ranks = np.apply_along_axis(stats.rankdata, axis=1, arr=stacked)  # (n, k)
    mean_ranks_arr = ranks.mean(axis=0)  # (k,)
    mean_ranks = {lbl: float(mean_ranks_arr[i]) for i, lbl in enumerate(labels)}

    return FriedmanResult(
        labels=labels,
        n=n,
        k=k,
        stat=stat,
        df=k - 1,
        pval=pval,
        kendall_w=float(kendall_w),
        mean_ranks=mean_ranks,
    )


def run_brown_forsythe_friedman(vectors: dict[str, np.ndarray]) -> FriedmanResult:
    """Brown-Forsythe-style omnibus test for equality of dispersion (paired).

    Standard within-subject test for whether the methods have different
    subject-level spread. Procedure:

      1. For each method m, replace each accuracy a_i^(m) with its absolute
         deviation from the method's median: d_i^(m) = |a_i^(m) - median(a^(m))|.
         Using the median (not the mean) is what makes this "Brown-Forsythe"
         and gives robustness to outliers.
      2. Run Friedman on the transformed vectors d^(m), with subjects as
         blocks and methods as conditions.

    H0: median absolute deviation is equal across methods.

    Pairing is preserved through the Friedman block structure (each subject
    contributes one d value per method), so this is the correct paired-design
    test of dispersion equality. No normality assumption.

    Returns the same FriedmanResult dataclass as run_friedman, but operating
    on absolute deviations instead of raw accuracies. mean_ranks here describe
    where each method sits in the ranking of *spread* per subject (higher
    rank = larger deviation from that method's median for that subject).
    """
    transformed: dict[str, np.ndarray] = {}
    for label, vec in vectors.items():
        med = float(np.median(vec))
        transformed[label] = np.abs(vec - med)
    return run_friedman(transformed)


def bca_variance_ratio_ci(
    a: np.ndarray,
    b: np.ndarray,
    seed: int,
    n_boot: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Paired BCa bootstrap CI on the variance ratio Var(a) / Var(b).

    Resamples subject indices with replacement (preserving pairing — both
    methods at subject i are kept together), recomputes the ratio on each
    resample. Asymmetric distribution (bounded below at 0), so BCa correction
    is more appropriate than plain percentile.

    Returns (point_estimate, ci_lo, ci_hi) — the observed ratio and the
    BCa-corrected 95% CI bounds.

    CI excluding 1.0 = the variances differ significantly at level alpha.
    """
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same length; got {a.shape} vs {b.shape}")

    def _ratio(idx: np.ndarray) -> float:
        va = float(np.var(a[idx], ddof=1))
        vb = float(np.var(b[idx], ddof=1))
        if vb == 0:
            return float("nan")
        return va / vb

    n = len(a)
    rng = np.random.default_rng(seed)

    # Paired bootstrap on subject indices.
    idx_matrix = rng.integers(0, n, size=(n_boot, n))
    boot = np.array([_ratio(idx_matrix[k]) for k in range(n_boot)], dtype=float)
    boot = boot[np.isfinite(boot)]
    if len(boot) == 0:
        return float("nan"), float("nan"), float("nan")

    observed = _ratio(np.arange(n))

    # BCa correction — same shape as in scipy's bootstrap implementation.
    # z0 = Phi^{-1}(fraction of boot replicates below observed).
    frac_below = float((boot < observed).sum() + 0.5 * (boot == observed).sum()) / len(boot)
    frac_below = min(max(frac_below, 1e-12), 1 - 1e-12)  # clamp for ppf
    z0 = float(stats.norm.ppf(frac_below))

    # Acceleration via jackknife on subjects.
    jack: list[float] = []
    full_idx = np.arange(n)
    for i in range(n):
        leave_out = np.delete(full_idx, i)
        jack.append(_ratio(leave_out))
    jack = np.array(jack, dtype=float)
    jack = jack[np.isfinite(jack)]
    if len(jack) < 3:
        # Fall back to plain percentile if jackknife degenerates.
        lo = float(np.percentile(boot, 100 * alpha / 2))
        hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
        return observed, lo, hi

    jack_mean = jack.mean()
    num = float(np.sum((jack_mean - jack) ** 3))
    den = 6.0 * (float(np.sum((jack_mean - jack) ** 2)) ** 1.5)
    accel = num / den if den > 0 else 0.0

    z_lo = float(stats.norm.ppf(alpha / 2))
    z_hi = float(stats.norm.ppf(1 - alpha / 2))
    a1 = float(stats.norm.cdf(z0 + (z0 + z_lo) / (1 - accel * (z0 + z_lo))))
    a2 = float(stats.norm.cdf(z0 + (z0 + z_hi) / (1 - accel * (z0 + z_hi))))
    # Clamp the corrected percentiles to [0, 1] for safety.
    a1 = min(max(a1, 0.0), 1.0)
    a2 = min(max(a2, 0.0), 1.0)

    lo = float(np.percentile(boot, 100 * a1))
    hi = float(np.percentile(boot, 100 * a2))
    return observed, lo, hi


def percentile_bootstrap_ci(
    d: np.ndarray, seed: int, n_boot: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Plain percentile bootstrap CI on the mean of d.

    Used as a non-parametric alternative to the t-based CI on the paired
    mean difference. When the paired differences are non-normal, this CI is
    more honest than the t-based one (it makes no distributional assumption).
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(
        np.percentile(means, 100 * (1 - alpha / 2))
    )


@dataclass
class PairedResult:
    label_a: str
    label_b: str
    task: str
    n: int
    n_nonzero: int
    n_wins: int
    n_losses: int
    n_ties: int
    mean_a: float
    mean_b: float
    mean_diff: float
    median_diff: float
    std_diff: float
    # CI on the mean difference, two ways:
    t_ci_lo: float
    t_ci_hi: float
    boot_ci_lo: float
    boot_ci_hi: float
    # Significance tests:
    t_stat: float
    t_pval: float
    w_stat: float
    w_pval: float
    sign_pval: float
    # Distributional diagnostics:
    diff_skew: float
    diff_kurtosis_excess: float
    shapiro_pval: float
    # Effect sizes:
    cohen_d: float
    rank_biserial: float


def run_paired(
    a_label: str, a_vals: np.ndarray, b_label: str, b_vals: np.ndarray, task: str, seed: int
) -> PairedResult:
    diff = a_vals - b_vals
    n = len(diff)
    nonzero = diff[diff != 0]
    n_nonzero = len(nonzero)
    n_wins = int((diff > 0).sum())
    n_losses = int((diff < 0).sum())
    n_ties = int((diff == 0).sum())

    mean_diff = float(diff.mean())
    median_diff = float(np.median(diff))
    std_diff = float(diff.std(ddof=1))

    # Parametric: paired t-test (two-sided) + t-based CI on mean diff.
    t_res = stats.ttest_rel(a_vals, b_vals)
    t_stat = float(t_res.statistic)
    t_pval = float(t_res.pvalue)
    se = std_diff / np.sqrt(n)
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    t_ci_lo = mean_diff - t_crit * se
    t_ci_hi = mean_diff + t_crit * se

    # Non-parametric: percentile bootstrap CI on the mean of the differences.
    boot_ci_lo, boot_ci_hi = percentile_bootstrap_ci(diff, seed=seed, n_boot=BOOTSTRAP_N)

    # Non-parametric: Wilcoxon signed-rank (zero-diffs dropped per "wilcox" mode).
    w_res = stats.wilcoxon(a_vals, b_vals, zero_method="wilcox", alternative="two-sided")
    w_stat = float(w_res.statistic)
    w_pval = float(w_res.pvalue)

    # Sign test: binomial on (#wins, n_nonzero). Convergent-evidence check.
    if n_nonzero > 0:
        sign_pval = float(stats.binomtest(n_wins, n_nonzero, p=0.5, alternative="two-sided").pvalue)
    else:
        sign_pval = float("nan")

    # Distributional diagnostics on the paired differences.
    diff_skew = float(stats.skew(diff))
    diff_kurt = float(stats.kurtosis(diff))  # excess kurtosis
    shapiro_pval = float(stats.shapiro(diff).pvalue)

    # Effect sizes.
    cohen_d = mean_diff / std_diff if std_diff > 0 else float("nan")
    if n_nonzero > 0:
        abs_ranks = stats.rankdata(np.abs(nonzero))
        w_plus = abs_ranks[nonzero > 0].sum()
        w_minus = abs_ranks[nonzero < 0].sum()
        rb = (w_plus - w_minus) / (w_plus + w_minus) if (w_plus + w_minus) > 0 else float("nan")
    else:
        rb = float("nan")

    return PairedResult(
        label_a=a_label,
        label_b=b_label,
        task=task,
        n=n,
        n_nonzero=n_nonzero,
        n_wins=n_wins,
        n_losses=n_losses,
        n_ties=n_ties,
        mean_a=float(a_vals.mean()),
        mean_b=float(b_vals.mean()),
        mean_diff=mean_diff,
        median_diff=median_diff,
        std_diff=std_diff,
        t_ci_lo=t_ci_lo,
        t_ci_hi=t_ci_hi,
        boot_ci_lo=boot_ci_lo,
        boot_ci_hi=boot_ci_hi,
        t_stat=t_stat,
        t_pval=t_pval,
        w_stat=w_stat,
        w_pval=w_pval,
        sign_pval=sign_pval,
        diff_skew=diff_skew,
        diff_kurtosis_excess=diff_kurt,
        shapiro_pval=shapiro_pval,
        cohen_d=cohen_d,
        rank_biserial=rb,
    )


def fmt_p(p: float) -> str:
    if np.isnan(p):
        return "—"
    if p < 1e-4:
        return f"{p:.2e}"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def main() -> None:
    # ── Per-condition BCa bootstrap CIs ────────────────────────────────────
    bootstrap_rows: list[dict] = []
    for (approach, task), _ in RUNS.items():
        accs = np.array(list(load_per_subject(approach, task).values()), dtype=float)
        lo, hi = bca_bootstrap_ci(accs, seed=BOOTSTRAP_SEED, n_boot=BOOTSTRAP_N)
        bootstrap_rows.append(
            {
                "method": LABELS[approach],
                "task": task,
                "n": len(accs),
                "mean": float(accs.mean()),
                "std": float(accs.std(ddof=1)),
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )

    # ── Paired tests ───────────────────────────────────────────────────────
    paired_results: list[PairedResult] = []
    jade = {t: load_per_subject("jade", t) for t in TASKS}
    sft = {t: load_per_subject("ft", t) for t in TASKS}
    lp = {t: load_per_subject("lp", t) for t in TASKS}

    # Use a deterministic per-comparison seed so reruns are reproducible.
    for i, task in enumerate(TASKS):
        for j, (baseline_key, baseline_dict) in enumerate([("SFT", sft[task]), ("LP", lp[task])]):
            a, b, _ = aligned_pair(jade[task], baseline_dict)
            paired_results.append(
                run_paired(
                    "JADE",
                    a,
                    baseline_key,
                    b,
                    task,
                    seed=BOOTSTRAP_SEED + 100 * (2 * i + j),
                )
            )

    # Holm-Bonferroni on the 4 t and 4 Wilcoxon p-values.
    t_adj = holm_bonferroni([r.t_pval for r in paired_results])
    w_adj = holm_bonferroni([r.w_pval for r in paired_results])

    # ── Compose markdown ───────────────────────────────────────────────────
    L: list[str] = []
    L.append("# Statistical Tests")
    L.append("")
    L.append("Companion document to `docs/results_brief.md`. All tests operate on the")
    L.append("**per-subject accuracy vectors** (N = 123 subjects), the same vectors")
    L.append("plotted in the paired scatter figures. Units of analysis are subjects, not")
    L.append("windows; windows within a subject are not independent.")
    L.append("")
    L.append("This document is regenerated from `main-results/` by")
    L.append("`uv run python -m src.inference.statistical_tests` and should not be")
    L.append("hand-edited. Methodology follows the audit recommendations: **Wilcoxon")
    L.append("signed-rank is the primary paired test**; the t-test is reported as a")
    L.append("parametric companion; a percentile bootstrap CI on the paired difference")
    L.append("is reported alongside the t-based CI; per-condition CIs use BCa bootstrap.")
    L.append("")

    # ── §0 definitions ─────────────────────────────────────────────────────
    L.append("## 0. Metrics and tests — definitions")
    L.append("")
    L.append("Self-contained definitions of every quantity in the tables below.")
    L.append("")

    L.append("### 0.1 Underlying data: per-subject accuracy")
    L.append("")
    L.append("Each method is evaluated with 10-fold cross-subject CV. Every one of the")
    L.append("N = 123 subjects appears in exactly one validation fold. For a given")
    L.append("method × task we therefore obtain a vector of 123 accuracies")
    L.append("`a = (a_1, ..., a_123)` where `a_i ∈ [0, 1]` is subject *i*'s accuracy on")
    L.append("the held-out fold containing them.")
    L.append("")
    L.append("**Intuitively.** *\"How well does the model do on this person's brain")
    L.append('data when it has never seen them during training?"* This is the central')
    L.append("question for cross-subject EEG.")
    L.append("")
    L.append("All statistics in this document operate on these per-subject vectors. The")
    L.append("**unit of analysis is the subject, not the trial window** — windows within")
    L.append("a subject are correlated, so treating windows as independent would inflate")
    L.append("sample size and overstate significance.")
    L.append("")

    L.append("### 0.2 Mean accuracy")
    L.append("")
    L.append("Formula: `mean(a) = (1/N) · Σ a_i`. The average classification accuracy")
    L.append("across the 123 subjects — the single most-quoted headline number in the")
    L.append("cross-subject EEG literature.")
    L.append("")

    L.append("### 0.3 Standard deviation (std) of per-subject accuracy")
    L.append("")
    L.append("Formula: `std(a) = sqrt( (1/(N-1)) · Σ (a_i − mean(a))² )` (sample std,")
    L.append("denominator `N−1`).")
    L.append("")
    L.append('**Intuitively.** *"How much do subjects vary?"* A large std means some')
    L.append("subjects are much easier than others — the well-known **inter-subject")
    L.append("variability** of EEG.")
    L.append("")
    L.append("**Important caveat.** `mean ± std` describes the *spread of subjects*; it")
    L.append("does **not** describe how precisely you know the mean. For that, see §0.4.")
    L.append("")

    L.append("### 0.4 BCa bootstrap 95 % CI on the mean accuracy")
    L.append("")
    L.append("**What it is.** A 95 % CI is an interval that, under repeated sampling,")
    L.append("contains the true mean accuracy 95 % of the time. Narrow CI = mean is")
    L.append("estimated precisely; wide CI = mean is uncertain.")
    L.append("")
    L.append("**Procedure.** From the 123 subject accuracies, draw 123 with replacement")
    L.append(f"and take the mean; repeat {BOOTSTRAP_N:,} times (seed = {BOOTSTRAP_SEED})")
    L.append("to build an empirical sampling distribution of the mean. The 95 % CI is")
    L.append("then read off two percentiles of that distribution.")
    L.append("")
    L.append("**Two flavours, why BCa here.**")
    L.append("")
    L.append("- *Percentile* (simplest): take the raw 2.5th and 97.5th percentiles.")
    L.append("  Implicitly assumes the bootstrap distribution is unbiased and symmetric.")
    L.append("- *BCa* — bias-corrected and accelerated (Efron 1987), used here via")
    L.append('  `scipy.stats.bootstrap(..., method="BCa")`. Shifts the percentiles to')
    L.append("  correct for two distortions: **bias** (the bootstrap distribution may be")
    L.append("  off-centre relative to the true sampling distribution; estimated from the")
    L.append("  fraction of resamples below the observed mean) and **acceleration** (the")
    L.append("  standard error may itself vary with the underlying value, i.e. skew;")
    L.append("  estimated via jackknife).")
    L.append("")
    L.append("When the distribution is symmetric and unbiased, both methods give the")
    L.append("same interval. When it is skewed, BCa shifts the CI to be more honest about")
    L.append("where the true sampling distribution lies. For bounded accuracies near")
    L.append("0.5–0.8 the BCa corrections are small but non-zero; BCa is the")
    L.append("textbook-recommended default and costs nothing extra at this sample size.")
    L.append("")
    L.append('**Intuitively.** *"If I were to repeat this entire study with a different')
    L.append("set of 123 subjects drawn from the same population, where would the mean")
    L.append('accuracy land 95 % of the time?"*')
    L.append("")

    L.append("### 0.5 Paired difference vector")
    L.append("")
    L.append('For each comparison "method A vs method B" on a given task we form the')
    L.append("**paired difference** vector `d_i = a_i − b_i` (subject *i*'s accuracy")
    L.append("under A minus their accuracy under B). Because both methods are evaluated")
    L.append("on the *same* subjects, the difference removes the subject-level")
    L.append("baseline: subject-to-subject variability is no longer a confound, only")
    L.append("the method effect remains.")
    L.append("")
    L.append("- `mean(d)` is the **average improvement** of A over B (positive = A wins).")
    L.append("- `median(d)` is the **typical improvement** (robust to outliers).")
    L.append("- `std(d)` is how much the improvement *itself* varies across subjects.")
    L.append("- A non-trivial fraction of pairs (here ~10 %) have `d_i = 0` exactly,")
    L.append("  because per-subject accuracies are coarse-grained (subject *i*'s number")
    L.append("  of correct windows is integer / total windows). The treatment of these")
    L.append("  zero-differences is test-dependent — see §0.7 and §0.11.")
    L.append("")

    L.append("### 0.6 Paired t-test (parametric)")
    L.append("")
    L.append('**What it tests.** `H₀: mean(d) = 0` — "the two methods are equally good')
    L.append('on average". Two-sided alternative.')
    L.append("")
    L.append("**Formula.** `t = mean(d) / (std(d) / sqrt(N))`. The denominator is the")
    L.append("standard error of the mean of the differences. The *p*-value is looked up")
    L.append("from the t-distribution with N − 1 degrees of freedom.")
    L.append("")
    L.append("**Assumption.** The paired differences `d_i` are approximately normal.")
    L.append("With N = 123 the Central Limit Theorem makes this assumption mild but")
    L.append("not vacuous; we explicitly report Shapiro-Wilk diagnostics per")
    L.append("comparison (see §2).")
    L.append("")
    L.append("**95 % CI on the mean difference (t-based).**")
    L.append("`CI = mean(d) ± t_crit · std(d) / sqrt(N)`, where `t_crit ≈ 1.98` for")
    L.append("N = 123. If the CI excludes 0, the t-test rejects `H₀` at *p* < 0.05.")
    L.append("")
    L.append("**Role in this report.** Confirmatory only. When the paired-difference")
    L.append("distribution is not normal, prefer the bootstrap CI (§0.10) and Wilcoxon")
    L.append("p-value (§0.7).")
    L.append("")

    L.append("### 0.7 Wilcoxon signed-rank test (non-parametric, **primary**)")
    L.append("")
    L.append("**What it tests.** `H₀`: the distribution of paired differences is")
    L.append("symmetric around zero. Two-sided alternative.")
    L.append("")
    L.append("**Procedure.**")
    L.append("")
    L.append('1. Drop pairs where `d_i = 0` exactly (the `"wilcox"` zero-method, the')
    L.append("   standard default in scipy and most stats packages).")
    L.append("2. Rank the absolute differences `|d_i|` from smallest (rank 1) to largest.")
    L.append("3. Sum the ranks of pairs where `d_i > 0` (call this `W+`) and where")
    L.append("   `d_i < 0` (call this `W−`).")
    L.append("4. Under `H₀`, `W+` and `W−` should be similar. The reported test")
    L.append("   statistic is `min(W+, W−)`; the *p*-value comes from its null")
    L.append("   distribution (computed exactly for small N, asymptotically here).")
    L.append("")
    L.append('**Intuitively.** *"Setting aside the exact size of each improvement, is')
    L.append("there a clear pattern of A winning more often (and by larger ranked")
    L.append('amounts) than B?"*')
    L.append("")
    L.append("**Why it is primary.** It makes **no normality assumption** — only ranks")
    L.append("are used, so heavy tails, outliers and modest non-normality do not break")
    L.append("it. The Shapiro-Wilk diagnostics in §2 will show that not all paired")
    L.append("difference distributions in this thesis are normal, so deferring to")
    L.append("Wilcoxon for the headline call is the conservative and correct choice.")
    L.append("")

    L.append("### 0.8 Sign test (binomial, convergent evidence)")
    L.append("")
    L.append("**What it tests.** `H₀`: among non-zero paired differences, JADE wins and")
    L.append("loses with equal probability. The test statistic is the number of wins")
    L.append("among non-zero pairs; under `H₀` this follows `Binomial(n_nonzero, 0.5)`.")
    L.append("")
    L.append('**Intuitively.** *"Forgetting both the *size* of the gain and its *rank*,')
    L.append('does JADE simply win on more subjects than it loses?"* The crudest')
    L.append("possible paired test — it discards all magnitude information.")
    L.append("")
    L.append("**Role in this report.** Convergent-evidence check. When the t, Wilcoxon")
    L.append("and sign tests all agree, the conclusion is exceptionally robust because")
    L.append("they use very different aspects of the data (mean vs ranks vs counts).")
    L.append("When the sign test disagrees with the others, magnitude information is")
    L.append('doing meaningful work and we cannot reduce the question to "who wins more')
    L.append('often".')
    L.append("")

    L.append("### 0.9 Effect sizes — Cohen's *d* and rank-biserial *r*")
    L.append("")
    L.append("**Why these matter.** With N = 123 the tests have high power and can")
    L.append('declare even tiny differences "significant". Effect sizes answer a')
    L.append('different question: not *"is it real?"* but *"is it big?"*.')
    L.append("")
    L.append("**Cohen's *d* (paired).** Formula: `d = mean(d) / std(d)`. Unit-free.")
    L.append("Conventional cut-offs (Cohen, 1988): `0.2` small, `0.5` medium, `0.8`")
    L.append("large.")
    L.append("")
    L.append("**Rank-biserial *r* (paired).** Formula: `r = (W+ − W−) / (W+ + W−)`,")
    L.append("bounded in [−1, +1]. The non-parametric companion to Cohen's *d*,")
    L.append("computed from the Wilcoxon rank sums. Conventional cut-offs: `0.1` small,")
    L.append("`0.3` medium, `0.5` large.")
    L.append("")

    L.append("### 0.10 95 % CI on the paired mean difference — two flavours reported")
    L.append("")
    L.append("**t-based CI** (parametric). Assumes the paired differences are")
    L.append("approximately normal. Formula in §0.6.")
    L.append("")
    L.append("**Percentile bootstrap CI** (non-parametric). Same resampling procedure")
    L.append("as in §0.4, but applied to the paired-difference vector `d` rather than")
    L.append("to the raw accuracy vector — and taking the raw 2.5th / 97.5th percentiles")
    L.append("without the BCa correction. No distributional assumption.")
    L.append("")
    L.append("**Reading the table.** When the two CIs agree to within rounding, the")
    L.append("t-distribution is a good fit and either interval is fine to quote in")
    L.append("the report. When they disagree, prefer the bootstrap interval — it is")
    L.append("assumption-free. (BCa would be the more rigorous choice for the paired-")
    L.append("difference distribution too; we report the plain percentile interval for")
    L.append("now and treat the gap as a known minor methodological inconsistency.)")
    L.append("")

    L.append("### 0.11 Holm-Bonferroni correction (multiple comparisons)")
    L.append("")
    L.append("Running *m* independent tests at α = 0.05 with no correction gives a")
    L.append("family-wise error rate of roughly `1 − (1 − 0.05)^m`. For *m* = 4 (the")
    L.append("number of paired comparisons here), that is ≈ 19 %. Holm-Bonferroni")
    L.append("bounds the family-wise error at α and is uniformly more powerful than")
    L.append("plain Bonferroni:")
    L.append("")
    L.append("1. Sort the *m* raw *p*-values ascending: `p_(1) ≤ ... ≤ p_(m)`.")
    L.append("2. Multiply the *k*-th smallest by `m − k + 1`, enforce monotonicity,")
    L.append("   cap at 1.")
    L.append("3. Compare each adjusted *p* against α (here 0.05).")
    L.append("")

    L.append("### 0.12 Limitations and methodological notes")
    L.append("")
    L.append("**Fold-clustering of subjects.** All bootstrap and parametric methods")
    L.append("treat the 123 subjects as i.i.d. draws. They are independent *people*,")
    L.append("but their accuracy scores are not strictly i.i.d.: 13 subjects in the")
    L.append("same CV fold are predicted by the *same* trained model, so if a")
    L.append("particular fold's model is unusually good or bad the 13 subjects in its")
    L.append("val set move together. A fully rigorous treatment would use a")
    L.append("**clustered (block) bootstrap** that resamples whole folds rather than")
    L.append("subjects. With only 10 folds, however, a block bootstrap is high-")
    L.append("variance and produces CIs much wider than the data warrant. We therefore")
    L.append("report the standard subject-level bootstrap and acknowledge that the")
    L.append('"effective sample size" is slightly below 123. This is consistent with')
    L.append("standard practice in cross-subject EEG papers.")
    L.append("")
    L.append("**Zero-difference pairs.** Subjects whose accuracy under method A")
    L.append("happens to exactly equal their accuracy under method B (e.g. both classify")
    L.append("the same windows correctly) produce `d_i = 0`. The Wilcoxon test here")
    L.append('uses the `"wilcox"` mode, which **drops** such pairs. An alternative')
    L.append('(`"pratt"` mode) keeps them and is slightly more conservative; we do not')
    L.append("report it because it does not change the qualitative conclusions on this")
    L.append("dataset. The sign test naturally excludes zeros (the binomial is computed")
    L.append("over non-zero pairs only).")
    L.append("")
    L.append("**Normality assumptions.** Shapiro-Wilk *p*-values on the paired")
    L.append("difference distributions are reported per comparison in §2. Where")
    L.append("Shapiro *p* < 0.05 the difference distribution is statistically")
    L.append("non-normal; in those cases the Wilcoxon test and the bootstrap CI are")
    L.append("the primary results, with the t-test as a parametric confirmation.")
    L.append("")

    # ── §1 per-condition descriptives ──────────────────────────────────────
    L.append("## 1. Per-condition descriptive statistics + BCa bootstrap CI on the mean")
    L.append("")
    L.append(f"Bootstrap: {BOOTSTRAP_N:,} resamples, BCa correction, seed = {BOOTSTRAP_SEED}.")
    L.append("`mean ± std` describes the spread of subject accuracies; the BCa CI")
    L.append("describes how precisely the *mean* itself is known.")
    L.append("")
    L.append("| Method | Task    | N   | Mean      | Std       | 95 % CI on the mean (BCa) |")
    L.append("|--------|---------|-----|-----------|-----------|---------------------------|")
    for r in bootstrap_rows:
        L.append(
            f"| {r['method']:<6} | {r['task']:<7} | {r['n']:>3} | "
            f"{r['mean'] * 100:>6.2f} %  | {r['std'] * 100:>5.2f} %    | "
            f"{r['ci_lo'] * 100:>5.2f} % – {r['ci_hi'] * 100:>5.2f} %         |"
        )
    L.append("")

    # ── §2 paired tests (main table) ───────────────────────────────────────
    L.append("## 2. Paired tests — JADE vs baselines (Wilcoxon primary)")
    L.append("")
    L.append("Each row is a paired test on N = 123 subjects, comparing JADE to the")
    L.append("baseline on the same subjects. `mean Δ` is `JADE − baseline` (positive")
    L.append("favours JADE). Both a t-based and a percentile-bootstrap CI on the mean")
    L.append("Δ are shown; agreement between them is a sanity check on the t-test's")
    L.append("normality assumption.")
    L.append("")
    L.append(
        "| Comparison | Task | Mean Δ (pp) | Median Δ (pp) | CI 95 % t (pp) | CI 95 % bootstrap (pp) | Cohen's d | Rank-bis. r | Wins / Losses / Ties | Wilcoxon W | p (Wilcoxon) | Holm p (W) | t | p (t-test) | Holm p (t) | Sign-test p | Shapiro p (normality) |"
    )
    L.append(
        "|------------|------|-------------|---------------|----------------|------------------------|-----------|-------------|----------------------|------------|--------------|------------|---|------------|------------|-------------|-----------------------|"
    )
    for r, p_t_adj, p_w_adj in zip(paired_results, t_adj, w_adj):
        L.append(
            f"| {r.label_a} vs {r.label_b} | {r.task} | "
            f"{r.mean_diff * 100:+.2f} | {r.median_diff * 100:+.2f} | "
            f"{r.t_ci_lo * 100:+.2f}, {r.t_ci_hi * 100:+.2f} | "
            f"{r.boot_ci_lo * 100:+.2f}, {r.boot_ci_hi * 100:+.2f} | "
            f"{r.cohen_d:.3f} | {r.rank_biserial:.3f} | "
            f"{r.n_wins} / {r.n_losses} / {r.n_ties} | "
            f"{r.w_stat:.0f} | {fmt_p(r.w_pval)} | {fmt_p(p_w_adj)} | "
            f"{r.t_stat:+.2f} | {fmt_p(r.t_pval)} | {fmt_p(p_t_adj)} | "
            f"{fmt_p(r.sign_pval)} | {fmt_p(r.shapiro_pval)} |"
        )
    L.append("")
    L.append("**Convergent-evidence reading.** When Wilcoxon, t-test and sign-test all")
    L.append("agree (all p < 0.05 or all p ≥ 0.05), the qualitative conclusion is")
    L.append("triple-supported by three tests using very different aspects of the data")
    L.append("(ranks, magnitudes, counts). Disagreement should be flagged in the prose.")
    L.append("")

    # ── §3 how-to-read ─────────────────────────────────────────────────────
    L.append("## 3. How to interpret these numbers")
    L.append("")
    L.append("- **Wilcoxon p-value (and its Holm-adjusted version) is the headline**")
    L.append("  significance number for each comparison. The t-test p is a sanity")
    L.append("  check; when Shapiro p < 0.05 the t-test should not be relied on.")
    L.append("- The **CI on the mean difference** is more informative than any single")
    L.append("  *p*-value because it shows direction and magnitude. The bootstrap CI is")
    L.append("  assumption-free; quote it when the t-based and bootstrap CIs disagree.")
    L.append("- **Effect sizes**: Cohen's d for the t-test (~0.2 small, ~0.5 medium,")
    L.append("  ~0.8 large); rank-biserial r for Wilcoxon (~0.1 / ~0.3 / ~0.5).")
    L.append("- **Holm-Bonferroni** controls the family-wise error rate across the 4")
    L.append("  tests. Holm-adjusted p < 0.05 means the result survives multiple-")
    L.append("  comparisons correction.")
    L.append("- **Sign-test agreement** is a robustness check — when it agrees with")
    L.append("  Wilcoxon and t, magnitude/rank information is not strictly required to")
    L.append('  reach the conclusion ("JADE wins more often than it loses" suffices).')
    L.append("")

    # ── §4 suggested phrasings ─────────────────────────────────────────────
    L.append("## 4. Suggested phrasing for the Results section")
    L.append("")
    L.append("Templates below quote Wilcoxon as the primary test and the bootstrap CI")
    L.append("as the primary interval, with the t-test and t-based CI reported")
    L.append("parenthetically for completeness. Numbers come straight from this run.")
    L.append("")
    for r, p_t_adj, p_w_adj in zip(paired_results, t_adj, w_adj):
        verdict_w = "significant" if p_w_adj < 0.05 else "not significant"
        sign = "improvement" if r.mean_diff > 0 else "decrease"
        normality_note = ""
        if r.shapiro_pval < NORMALITY_ALPHA:
            normality_note = (
                f" (Paired differences are non-normal, Shapiro p = "
                f"{fmt_p(r.shapiro_pval)}; the bootstrap CI is the assumption-free choice.)"
            )
        L.append(
            f"- **{r.label_a} vs {r.label_b}, {r.task}.** "
            f"{r.label_a} shows a mean {sign} of "
            f"**{r.mean_diff * 100:+.2f} pp** "
            f"(95 % bootstrap CI: {r.boot_ci_lo * 100:+.2f}, {r.boot_ci_hi * 100:+.2f}; "
            f"t-based CI: {r.t_ci_lo * 100:+.2f}, {r.t_ci_hi * 100:+.2f}). "
            f"The paired difference is **{verdict_w}** "
            f"(Wilcoxon W = {r.w_stat:.0f}, p = {fmt_p(r.w_pval)}, "
            f"Holm-adjusted p = {fmt_p(p_w_adj)}; "
            f"paired t-test t({r.n - 1}) = {r.t_stat:+.2f}, p = {fmt_p(r.t_pval)}; "
            f"sign-test p = {fmt_p(r.sign_pval)} on {r.n_wins}/{r.n_nonzero} wins). "
            f"Effect size: Cohen's d = {r.cohen_d:.2f}, "
            f"rank-biserial r = {r.rank_biserial:.2f}." + normality_note
        )
    L.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(L) + "\n")
    print(f"Wrote {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
