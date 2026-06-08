# Statistical Tests — Generalization

Companion to `docs/statistical_tests.md`, applied to the FACED
stimulus-generalization runs (`main-results/*_generalization/*_gen_avg.json`,
produced by `src.inference.average_gen_seeds`). The methodology differs
from the main document because the generalization question is omnibus:
*"do the three methods differ at all under held-out stimuli?"* — not
*"is JADE better than each baseline?"*

Protocol per task:

1. **Friedman omnibus** across LP, SFT, JADE — primary test on the *means*.
2. **Per-seed Friedman** on each gen seed's individual JSON — robustness check.
3. **Pairwise Wilcoxon signed-rank** on all method pairs, *only if* the
   mean omnibus is significant (α = 0.05), with Holm-Bonferroni across the
   pairs of that task.
4. **BCa bootstrap CIs** on the per-condition mean — identical helper
   to `statistical_tests.py`, so numbers are directly comparable.
5. **Dispersion analysis**: Brown-Forsythe-via-Friedman omnibus on the
   subject-level absolute deviations + pairwise BCa CIs on the
   variance ratio `Var(a)/Var(b)`. Tests whether the methods differ in
   per-subject consistency, independently of mean accuracy.

Family-wise correction across **tasks** (n=2) is applied to the omnibus
p-values themselves, since each task is one omnibus test in the family.

## 0. Definitions

**Friedman test.** Non-parametric repeated-measures ANOVA. Each subject
is a block; the k methods are conditions; ranks are assigned per subject
(rank 1 = lowest accuracy for that subject). Under H₀ the average rank
of each method equals (k+1)/2. The test statistic is
`χ² = 12 / (n·k·(k+1)) · Σ R_j² − 3·n·(k+1)`, distributed χ²(k−1) under H₀.

**Kendall's W.** Effect size for Friedman, `W = χ² / (n·(k−1))`. Bounded
in [0, 1]: 0 = no agreement among subjects on method ranking, 1 = perfect
agreement (all subjects rank the methods identically).

**Per-seed Friedman.** The seed-averaged accuracies smooth over noise
between the two gen seeds. As a robustness check we rerun Friedman on
each seed's vectors individually; if both per-seed tests agree with the
seed-averaged test, the conclusion is not a smoothing artefact.

**Pairwise post-hoc Wilcoxon.** Identical to the §2 paired tests in
`statistical_tests.md`. Only computed when Friedman rejects H₀, to avoid
uncontrolled multiple-comparison fishing.

**Brown-Forsythe-via-Friedman dispersion omnibus.** For each method `m`,
each subject's accuracy `a_i^(m)` is replaced by its absolute deviation
from the method's median: `d_i^(m) = |a_i^(m) − median(a^(m))|`. Friedman
is then applied to the `d` vectors. Tests `H₀`: median absolute deviation
is equal across methods. The median centring (not mean) makes the test
robust to extreme subjects (Brown-Forsythe variant of Levene's test); the
Friedman wrapper preserves the within-subject pairing.

**Variance-ratio BCa CI.** Paired bootstrap CI on `Var(method_a) /
Var(method_b)`, with subject indices resampled with replacement so the
pairing is preserved. BCa correction is appropriate because the ratio
distribution is bounded below at 0 and asymmetric. A CI excluding 1.0
means the variances differ significantly at the corresponding level.
Variance ratio < 1 means method `a` has tighter subject-level spread
(more consistent across subjects) than method `b`. The CI is reported
regardless of the dispersion-omnibus outcome — it is an effect-size
measure; significance claims still defer to the omnibus.

## Task: 9-class

Methods available: **LP, SFT, JADE** (N = 123 subjects).

### Per-condition BCa CI on the mean accuracy

| Method | N | Mean | Std | 95 % CI on the mean (BCa) |
|--------|---|------|-----|---------------------------|
| LP     | 123 |  16.08 %  |  5.77 %    | 15.13 % – 17.18 %         |
| SFT    | 123 |  15.82 %  |  5.04 %    | 14.98 % – 16.74 %         |
| JADE   | 123 |  15.88 %  |  4.72 %    | 15.07 % – 16.74 %         |

### Friedman omnibus test

- χ²(2) = **0.259**, p = **0.878** (Holm across tasks, n=2: p = 1.000)
- Kendall's W = **0.0011** (effect size in [0, 1])
- Mean ranks (higher rank = better method on a given subject):
    - LP: 2.028
    - SFT: 2.004
    - JADE: 1.967

**Fail to reject H₀**: no significant difference among the three methods.

### Per-seed Friedman (robustness check)

| Seed | χ² | df | p (raw) | Kendall's W | Verdict |
|------|-----|----|---------|-------------|---------|
| 123 | 1.314 | 2 | 0.518 | 0.0053 | fail to reject |
| 789 | 0.217 | 2 | 0.897 | 0.0009 | fail to reject |

### Pairwise post-hoc

Omnibus not significant (Friedman p ≥ 0.05) — **post-hoc tests not run** to avoid uncontrolled multiple-comparison fishing.

### Dispersion omnibus (Brown-Forsythe via Friedman)

- χ²(2) = **1.757**, p = **0.415**
- Kendall's W = **0.0071**
- Mean ranks of |a − median(a)| per method:
    - LP: 2.085
    - SFT: 1.935
    - JADE: 1.980

**Fail to reject H₀**: no significant difference in dispersion.

### Pairwise variance ratios (BCa CIs)

Variance ratio < 1 ⇒ method `a` has tighter subject-level spread than method `b`. CI excluding 1.0 ⇒ the difference is significant at the corresponding level. The Holm-adjusted p column treats the variance-ratio CIs as a family of tests across pairs of this task.

| Comparison (a vs b) | Var(a) | Var(b) | Ratio Var(a)/Var(b) | 95 % BCa CI | approx p | Holm p |
|----------------------|--------|--------|---------------------|-------------|----------|--------|
| LP vs SFT | 0.00332 | 0.00254 | **1.307** | [0.911, 1.758] | 0.154 | 0.309 |
| LP vs JADE | 0.00332 | 0.00223 | **1.493** | [1.131, 1.976] | 0.022 | 0.066 |
| SFT vs JADE | 0.00254 | 0.00223 | **1.142** | [0.856, 1.573] | 0.438 | 0.438 |

## Task: binary

Methods available: **LP, SFT, JADE** (N = 123 subjects).

### Per-condition BCa CI on the mean accuracy

| Method | N | Mean | Std | 95 % CI on the mean (BCa) |
|--------|---|------|-----|---------------------------|
| LP     | 123 |  58.82 %  |  9.41 %    | 57.18 % – 60.47 %         |
| SFT    | 123 |  59.33 %  |  9.30 %    | 57.72 % – 61.01 %         |
| JADE   | 123 |  59.55 %  |  7.79 %    | 58.18 % – 60.91 %         |

### Friedman omnibus test

- χ²(2) = **1.208**, p = **0.547** (Holm across tasks, n=2: p = 1.000)
- Kendall's W = **0.0049** (effect size in [0, 1])
- Mean ranks (higher rank = better method on a given subject):
    - LP: 1.923
    - SFT: 2.045
    - JADE: 2.033

**Fail to reject H₀**: no significant difference among the three methods.

### Per-seed Friedman (robustness check)

| Seed | χ² | df | p (raw) | Kendall's W | Verdict |
|------|-----|----|---------|-------------|---------|
| 123 | 0.455 | 2 | 0.797 | 0.0018 | fail to reject |
| 789 | 1.840 | 2 | 0.399 | 0.0075 | fail to reject |

### Pairwise post-hoc

Omnibus not significant (Friedman p ≥ 0.05) — **post-hoc tests not run** to avoid uncontrolled multiple-comparison fishing.

### Dispersion omnibus (Brown-Forsythe via Friedman)

- χ²(2) = **4.234**, p = **0.120**
- Kendall's W = **0.0172**
- Mean ranks of |a − median(a)| per method:
    - LP: 2.053
    - SFT: 2.093
    - JADE: 1.854

**Fail to reject H₀**: no significant difference in dispersion.

### Pairwise variance ratios (BCa CIs)

Variance ratio < 1 ⇒ method `a` has tighter subject-level spread than method `b`. CI excluding 1.0 ⇒ the difference is significant at the corresponding level. The Holm-adjusted p column treats the variance-ratio CIs as a family of tests across pairs of this task.

| Comparison (a vs b) | Var(a) | Var(b) | Ratio Var(a)/Var(b) | 95 % BCa CI | approx p | Holm p |
|----------------------|--------|--------|---------------------|-------------|----------|--------|
| LP vs SFT | 0.00886 | 0.00865 | **1.024** | [0.772, 1.323] | 0.862 | 0.862 |
| LP vs JADE | 0.00886 | 0.00607 | **1.460** | [1.095, 1.867] | 0.020 | 0.050 |
| SFT vs JADE | 0.00865 | 0.00607 | **1.425** | [1.115, 1.812] | 0.017 | 0.050 |

