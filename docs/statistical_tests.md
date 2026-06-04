# Statistical Tests

Companion document to `docs/results_brief.md`. All tests operate on the
**per-subject accuracy vectors** (N = 123 subjects), the same vectors
plotted in the paired scatter figures. Units of analysis are subjects, not
windows; windows within a subject are not independent.

This document is regenerated from `main-results/` by
`uv run python -m src.inference.statistical_tests` and should not be
hand-edited. Methodology follows the audit recommendations: **Wilcoxon
signed-rank is the primary paired test**; the t-test is reported as a
parametric companion; a percentile bootstrap CI on the paired difference
is reported alongside the t-based CI; per-condition CIs use BCa bootstrap.

## 0. Metrics and tests — definitions

Self-contained definitions of every quantity in the tables below.

### 0.1 Underlying data: per-subject accuracy

Each method is evaluated with 10-fold cross-subject CV. Every one of the
N = 123 subjects appears in exactly one validation fold. For a given
method × task we therefore obtain a vector of 123 accuracies
`a = (a_1, ..., a_123)` where `a_i ∈ [0, 1]` is subject *i*'s accuracy on
the held-out fold containing them.

**Intuitively.** *"How well does the model do on this person's brain
data when it has never seen them during training?"* This is the central
question for cross-subject EEG.

All statistics in this document operate on these per-subject vectors. The
**unit of analysis is the subject, not the trial window** — windows within
a subject are correlated, so treating windows as independent would inflate
sample size and overstate significance.

### 0.2 Mean accuracy

Formula: `mean(a) = (1/N) · Σ a_i`. The average classification accuracy
across the 123 subjects — the single most-quoted headline number in the
cross-subject EEG literature.

### 0.3 Standard deviation (std) of per-subject accuracy

Formula: `std(a) = sqrt( (1/(N-1)) · Σ (a_i − mean(a))² )` (sample std,
denominator `N−1`).

**Intuitively.** *"How much do subjects vary?"* A large std means some
subjects are much easier than others — the well-known **inter-subject
variability** of EEG.

**Important caveat.** `mean ± std` describes the *spread of subjects*; it
does **not** describe how precisely you know the mean. For that, see §0.4.

### 0.4 BCa bootstrap 95 % CI on the mean accuracy

**What it is.** A 95 % CI is an interval that, under repeated sampling,
contains the true mean accuracy 95 % of the time. Narrow CI = mean is
estimated precisely; wide CI = mean is uncertain.

**Procedure.** From the 123 subject accuracies, draw 123 with replacement
and take the mean; repeat 10,000 times (seed = 42)
to build an empirical sampling distribution of the mean. The 95 % CI is
then read off two percentiles of that distribution.

**Two flavours, why BCa here.**

- *Percentile* (simplest): take the raw 2.5th and 97.5th percentiles.
  Implicitly assumes the bootstrap distribution is unbiased and symmetric.
- *BCa* — bias-corrected and accelerated (Efron 1987), used here via
  `scipy.stats.bootstrap(..., method="BCa")`. Shifts the percentiles to
  correct for two distortions: **bias** (the bootstrap distribution may be
  off-centre relative to the true sampling distribution; estimated from the
  fraction of resamples below the observed mean) and **acceleration** (the
  standard error may itself vary with the underlying value, i.e. skew;
  estimated via jackknife).

When the distribution is symmetric and unbiased, both methods give the
same interval. When it is skewed, BCa shifts the CI to be more honest about
where the true sampling distribution lies. For bounded accuracies near
0.5–0.8 the BCa corrections are small but non-zero; BCa is the
textbook-recommended default and costs nothing extra at this sample size.

**Intuitively.** *"If I were to repeat this entire study with a different
set of 123 subjects drawn from the same population, where would the mean
accuracy land 95 % of the time?"*

### 0.5 Paired difference vector

For each comparison "method A vs method B" on a given task we form the
**paired difference** vector `d_i = a_i − b_i` (subject *i*'s accuracy
under A minus their accuracy under B). Because both methods are evaluated
on the *same* subjects, the difference removes the subject-level
baseline: subject-to-subject variability is no longer a confound, only
the method effect remains.

- `mean(d)` is the **average improvement** of A over B (positive = A wins).
- `median(d)` is the **typical improvement** (robust to outliers).
- `std(d)` is how much the improvement *itself* varies across subjects.
- A non-trivial fraction of pairs (here ~10 %) have `d_i = 0` exactly,
  because per-subject accuracies are coarse-grained (subject *i*'s number
  of correct windows is integer / total windows). The treatment of these
  zero-differences is test-dependent — see §0.7 and §0.11.

### 0.6 Paired t-test (parametric)

**What it tests.** `H₀: mean(d) = 0` — "the two methods are equally good
on average". Two-sided alternative.

**Formula.** `t = mean(d) / (std(d) / sqrt(N))`. The denominator is the
standard error of the mean of the differences. The *p*-value is looked up
from the t-distribution with N − 1 degrees of freedom.

**Assumption.** The paired differences `d_i` are approximately normal.
With N = 123 the Central Limit Theorem makes this assumption mild but
not vacuous; we explicitly report Shapiro-Wilk diagnostics per
comparison (see §2).

**95 % CI on the mean difference (t-based).**
`CI = mean(d) ± t_crit · std(d) / sqrt(N)`, where `t_crit ≈ 1.98` for
N = 123. If the CI excludes 0, the t-test rejects `H₀` at *p* < 0.05.

**Role in this report.** Confirmatory only. When the paired-difference
distribution is not normal, prefer the bootstrap CI (§0.10) and Wilcoxon
p-value (§0.7).

### 0.7 Wilcoxon signed-rank test (non-parametric, **primary**)

**What it tests.** `H₀`: the distribution of paired differences is
symmetric around zero. Two-sided alternative.

**Procedure.**

1. Drop pairs where `d_i = 0` exactly (the `"wilcox"` zero-method, the
   standard default in scipy and most stats packages).
2. Rank the absolute differences `|d_i|` from smallest (rank 1) to largest.
3. Sum the ranks of pairs where `d_i > 0` (call this `W+`) and where
   `d_i < 0` (call this `W−`).
4. Under `H₀`, `W+` and `W−` should be similar. The reported test
   statistic is `min(W+, W−)`; the *p*-value comes from its null
   distribution (computed exactly for small N, asymptotically here).

**Intuitively.** *"Setting aside the exact size of each improvement, is
there a clear pattern of A winning more often (and by larger ranked
amounts) than B?"*

**Why it is primary.** It makes **no normality assumption** — only ranks
are used, so heavy tails, outliers and modest non-normality do not break
it. The Shapiro-Wilk diagnostics in §2 will show that not all paired
difference distributions in this thesis are normal, so deferring to
Wilcoxon for the headline call is the conservative and correct choice.

### 0.8 Sign test (binomial, convergent evidence)

**What it tests.** `H₀`: among non-zero paired differences, JADE wins and
loses with equal probability. The test statistic is the number of wins
among non-zero pairs; under `H₀` this follows `Binomial(n_nonzero, 0.5)`.

**Intuitively.** *"Forgetting both the *size* of the gain and its *rank*,
does JADE simply win on more subjects than it loses?"* The crudest
possible paired test — it discards all magnitude information.

**Role in this report.** Convergent-evidence check. When the t, Wilcoxon
and sign tests all agree, the conclusion is exceptionally robust because
they use very different aspects of the data (mean vs ranks vs counts).
When the sign test disagrees with the others, magnitude information is
doing meaningful work and we cannot reduce the question to "who wins more
often".

### 0.9 Effect sizes — Cohen's *d* and rank-biserial *r*

**Why these matter.** With N = 123 the tests have high power and can
declare even tiny differences "significant". Effect sizes answer a
different question: not *"is it real?"* but *"is it big?"*.

**Cohen's *d* (paired).** Formula: `d = mean(d) / std(d)`. Unit-free.
Conventional cut-offs (Cohen, 1988): `0.2` small, `0.5` medium, `0.8`
large.

**Rank-biserial *r* (paired).** Formula: `r = (W+ − W−) / (W+ + W−)`,
bounded in [−1, +1]. The non-parametric companion to Cohen's *d*,
computed from the Wilcoxon rank sums. Conventional cut-offs: `0.1` small,
`0.3` medium, `0.5` large.

### 0.10 95 % CI on the paired mean difference — two flavours reported

**t-based CI** (parametric). Assumes the paired differences are
approximately normal. Formula in §0.6.

**Percentile bootstrap CI** (non-parametric). Same resampling procedure
as in §0.4, but applied to the paired-difference vector `d` rather than
to the raw accuracy vector — and taking the raw 2.5th / 97.5th percentiles
without the BCa correction. No distributional assumption.

**Reading the table.** When the two CIs agree to within rounding, the
t-distribution is a good fit and either interval is fine to quote in
the report. When they disagree, prefer the bootstrap interval — it is
assumption-free. (BCa would be the more rigorous choice for the paired-
difference distribution too; we report the plain percentile interval for
now and treat the gap as a known minor methodological inconsistency.)

### 0.11 Holm-Bonferroni correction (multiple comparisons)

Running *m* independent tests at α = 0.05 with no correction gives a
family-wise error rate of roughly `1 − (1 − 0.05)^m`. For *m* = 4 (the
number of paired comparisons here), that is ≈ 19 %. Holm-Bonferroni
bounds the family-wise error at α and is uniformly more powerful than
plain Bonferroni:

1. Sort the *m* raw *p*-values ascending: `p_(1) ≤ ... ≤ p_(m)`.
2. Multiply the *k*-th smallest by `m − k + 1`, enforce monotonicity,
   cap at 1.
3. Compare each adjusted *p* against α (here 0.05).

### 0.12 Limitations and methodological notes

**Fold-clustering of subjects.** All bootstrap and parametric methods
treat the 123 subjects as i.i.d. draws. They are independent *people*,
but their accuracy scores are not strictly i.i.d.: 13 subjects in the
same CV fold are predicted by the *same* trained model, so if a
particular fold's model is unusually good or bad the 13 subjects in its
val set move together. A fully rigorous treatment would use a
**clustered (block) bootstrap** that resamples whole folds rather than
subjects. With only 10 folds, however, a block bootstrap is high-
variance and produces CIs much wider than the data warrant. We therefore
report the standard subject-level bootstrap and acknowledge that the
"effective sample size" is slightly below 123. This is consistent with
standard practice in cross-subject EEG papers.

**Zero-difference pairs.** Subjects whose accuracy under method A
happens to exactly equal their accuracy under method B (e.g. both classify
the same windows correctly) produce `d_i = 0`. The Wilcoxon test here
uses the `"wilcox"` mode, which **drops** such pairs. An alternative
(`"pratt"` mode) keeps them and is slightly more conservative; we do not
report it because it does not change the qualitative conclusions on this
dataset. The sign test naturally excludes zeros (the binomial is computed
over non-zero pairs only).

**Normality assumptions.** Shapiro-Wilk *p*-values on the paired
difference distributions are reported per comparison in §2. Where
Shapiro *p* < 0.05 the difference distribution is statistically
non-normal; in those cases the Wilcoxon test and the bootstrap CI are
the primary results, with the t-test as a parametric confirmation.

## 1. Per-condition descriptive statistics + BCa bootstrap CI on the mean

Bootstrap: 10,000 resamples, BCa correction, seed = 42.
`mean ± std` describes the spread of subject accuracies; the BCa CI
describes how precisely the *mean* itself is known.

| Method | Task    | N   | Mean      | Std       | 95 % CI on the mean (BCa) |
|--------|---------|-----|-----------|-----------|---------------------------|
| LP     | 9-class | 123 |  50.27 %  | 13.69 %    | 47.87 % – 52.70 %         |
| LP     | binary  | 123 |  71.64 %  |  9.29 %    | 69.99 % – 73.25 %         |
| SFT    | 9-class | 123 |  58.52 %  | 14.01 %    | 56.02 % – 60.95 %         |
| SFT    | binary  | 123 |  75.52 %  |  7.76 %    | 74.13 % – 76.89 %         |
| JADE   | 9-class | 123 |  62.03 %  | 14.70 %    | 59.35 % – 64.49 %         |
| JADE   | binary  | 123 |  76.32 %  |  8.12 %    | 74.86 % – 77.73 %         |

## 2. Paired tests — JADE vs baselines (Wilcoxon primary)

Each row is a paired test on N = 123 subjects, comparing JADE to the
baseline on the same subjects. `mean Δ` is `JADE − baseline` (positive
favours JADE). Both a t-based and a percentile-bootstrap CI on the mean
Δ are shown; agreement between them is a sanity check on the t-test's
normality assumption.

| Comparison | Task | Mean Δ (pp) | Median Δ (pp) | CI 95 % t (pp) | CI 95 % bootstrap (pp) | Cohen's d | Rank-bis. r | Wins / Losses / Ties | Wilcoxon W | p (Wilcoxon) | Holm p (W) | t | p (t-test) | Holm p (t) | Sign-test p | Shapiro p (normality) |
|------------|------|-------------|---------------|----------------|------------------------|-----------|-------------|----------------------|------------|--------------|------------|---|------------|------------|-------------|-----------------------|
| JADE vs SFT | 9-class | +3.51 | +3.57 | +2.60, +4.42 | +2.60, +4.44 | 0.689 | 0.704 | 88 / 22 / 13 | 904 | 1.45e-10 | 4.36e-10 | +7.64 | 5.40e-12 | 1.62e-11 | 1.55e-10 | 0.646 |
| JADE vs LP | 9-class | +11.76 | +10.71 | +10.23, +13.29 | +10.26, +13.29 | 1.375 | 0.968 | 113 / 6 / 4 | 114 | 4.87e-20 | 1.95e-19 | +15.25 | 4.81e-30 | 1.92e-29 | 1.10e-26 | 0.021 |
| JADE vs SFT | binary | +0.80 | +0.00 | -0.01, +1.62 | +0.00, +1.59 | 0.176 | 0.192 | 61 / 49 / 13 | 2466 | 0.080 | 0.080 | +1.95 | 0.053 | 0.053 | 0.294 | 0.069 |
| JADE vs LP | binary | +4.69 | +4.17 | +3.46, +5.91 | +3.49, +5.88 | 0.684 | 0.671 | 91 / 26 / 6 | 1134 | 2.86e-10 | 5.72e-10 | +7.59 | 7.18e-12 | 1.62e-11 | 1.21e-09 | 0.420 |

**Convergent-evidence reading.** When Wilcoxon, t-test and sign-test all
agree (all p < 0.05 or all p ≥ 0.05), the qualitative conclusion is
triple-supported by three tests using very different aspects of the data
(ranks, magnitudes, counts). Disagreement should be flagged in the prose.

## 3. How to interpret these numbers

- **Wilcoxon p-value (and its Holm-adjusted version) is the headline**
  significance number for each comparison. The t-test p is a sanity
  check; when Shapiro p < 0.05 the t-test should not be relied on.
- The **CI on the mean difference** is more informative than any single
  *p*-value because it shows direction and magnitude. The bootstrap CI is
  assumption-free; quote it when the t-based and bootstrap CIs disagree.
- **Effect sizes**: Cohen's d for the t-test (~0.2 small, ~0.5 medium,
  ~0.8 large); rank-biserial r for Wilcoxon (~0.1 / ~0.3 / ~0.5).
- **Holm-Bonferroni** controls the family-wise error rate across the 4
  tests. Holm-adjusted p < 0.05 means the result survives multiple-
  comparisons correction.
- **Sign-test agreement** is a robustness check — when it agrees with
  Wilcoxon and t, magnitude/rank information is not strictly required to
  reach the conclusion ("JADE wins more often than it loses" suffices).

## 4. Suggested phrasing for the Results section

Templates below quote Wilcoxon as the primary test and the bootstrap CI
as the primary interval, with the t-test and t-based CI reported
parenthetically for completeness. Numbers come straight from this run.

- **JADE vs SFT, 9-class.** JADE shows a mean improvement of **+3.51 pp** (95 % bootstrap CI: +2.60, +4.44; t-based CI: +2.60, +4.42). The paired difference is **significant** (Wilcoxon W = 904, p = 1.45e-10, Holm-adjusted p = 4.36e-10; paired t-test t(122) = +7.64, p = 5.40e-12; sign-test p = 1.55e-10 on 88/110 wins). Effect size: Cohen's d = 0.69, rank-biserial r = 0.70.
- **JADE vs LP, 9-class.** JADE shows a mean improvement of **+11.76 pp** (95 % bootstrap CI: +10.26, +13.29; t-based CI: +10.23, +13.29). The paired difference is **significant** (Wilcoxon W = 114, p = 4.87e-20, Holm-adjusted p = 1.95e-19; paired t-test t(122) = +15.25, p = 4.81e-30; sign-test p = 1.10e-26 on 113/119 wins). Effect size: Cohen's d = 1.38, rank-biserial r = 0.97. (Paired differences are non-normal, Shapiro p = 0.021; the bootstrap CI is the assumption-free choice.)
- **JADE vs SFT, binary.** JADE shows a mean improvement of **+0.80 pp** (95 % bootstrap CI: +0.00, +1.59; t-based CI: -0.01, +1.62). The paired difference is **not significant** (Wilcoxon W = 2466, p = 0.080, Holm-adjusted p = 0.080; paired t-test t(122) = +1.95, p = 0.053; sign-test p = 0.294 on 61/110 wins). Effect size: Cohen's d = 0.18, rank-biserial r = 0.19.
- **JADE vs LP, binary.** JADE shows a mean improvement of **+4.69 pp** (95 % bootstrap CI: +3.49, +5.88; t-based CI: +3.46, +5.91). The paired difference is **significant** (Wilcoxon W = 1134, p = 2.86e-10, Holm-adjusted p = 5.72e-10; paired t-test t(122) = +7.59, p = 7.18e-12; sign-test p = 1.21e-09 on 91/117 wins). Effect size: Cohen's d = 0.68, rank-biserial r = 0.67.

