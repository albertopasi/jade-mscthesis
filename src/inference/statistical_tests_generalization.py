"""statistical_tests_generalization.py — Significance tests on generalization results.

Companion to `statistical_tests.py`, but for the FACED stimulus-generalization
runs. Methodology differs from the main results:

  * Primary question is *omnibus*: "do LP, SFT, JADE differ at all on this
    held-out-stimulus task?" — Friedman test across the three methods.
  * **Only if Friedman is significant** do we run pairwise post-hoc tests
    (Wilcoxon signed-rank on all three pairs: LP-SFT, LP-JADE, SFT-JADE),
    with Holm-Bonferroni across the 3 pairs.
  * Per-condition BCa bootstrap CIs on the mean accuracy, same as §1 of the
    main statistical_tests.md.
  * Per-seed Friedman robustness check: rerun the omnibus on each seed's
    individual `_gen_s{seed}.json` to confirm the conclusion is not a
    seed-averaging artefact.

Input files: `main-results/<approach>_<task>_generalization/<stem>_gen_avg.json`,
which are produced by `src.inference.average_gen_seeds`.

If a method is missing for a task, that task
is reported with whatever data is available — Friedman requires k>=3, so a
two-method task falls back to Wilcoxon, and a single-method task is skipped.

Writes a single markdown document at docs/statistical_tests_generalization.md.

Usage:
    uv run python -m src.inference.statistical_tests_generalization
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Reuse the helpers and constants already in statistical_tests.py so the two
# documents stay consistent (same BCa CI definition, same Holm correction).
from src.inference.statistical_tests import (
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    LABELS,
    FriedmanResult,
    bca_bootstrap_ci,
    bca_variance_ratio_ci,
    fmt_p,
    holm_bonferroni,
    run_brown_forsythe_friedman,
    run_friedman,
    run_paired,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "main-results"
OUT_PATH = PROJECT_ROOT / "docs" / "statistical_tests_generalization.md"

APPROACHES = ["lp", "ft", "jade"]
TASKS = ["9-class", "binary"]

# Family-wise alpha used to gate the post-hoc tests on the Friedman result.
OMNIBUS_ALPHA = 0.05


# ── I/O ────────────────────────────────────────────────────────────────────


def gen_folder(approach: str, task: str) -> Path:
    return RESULTS_ROOT / f"{approach}_{task}_generalization"


def find_gen_avg(approach: str, task: str) -> Path | None:
    """Find the `<stem>_gen_avg.json` in the generalization folder, if present.

    Returns None when the folder does not exist or has no `_gen_avg.json` —
    e.g. LP binary gen jobs not finished yet.
    """
    folder = gen_folder(approach, task)
    if not folder.exists():
        return None
    matches = sorted(folder.glob("*_gen_avg.json"))
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple _gen_avg.json files in {folder}: {[m.name for m in matches]}. "
            "Disambiguate before running the test."
        )
    return matches[0]


def find_per_seed_jsons(approach: str, task: str) -> dict[int, Path]:
    """Return {seed: path} for the per-seed JSONs in a generalization folder."""
    import re

    seed_re = re.compile(r"_gen_s(\d+)\.json$")
    folder = gen_folder(approach, task)
    if not folder.exists():
        return {}
    out: dict[int, Path] = {}
    for path in sorted(folder.glob("*_gen_s*.json")):
        m = seed_re.search(path.name)
        if m:
            out[int(m.group(1))] = path
    return out


def load_per_subject(path: Path) -> dict[int, float]:
    data = json.loads(path.read_text())
    return {int(k): float(v) for k, v in data["per_subject_acc"].items()}


def align_three(
    accs: dict[str, dict[int, float]],
) -> tuple[dict[str, np.ndarray], list[int]]:
    """Align all subjects across method dicts by intersection."""
    keys = list(accs.keys())
    common = sorted(set.intersection(*[set(accs[k].keys()) for k in keys]))
    if not common:
        raise RuntimeError(f"No overlapping subjects across {keys}")
    return (
        {k: np.array([accs[k][s] for s in common], dtype=float) for k in keys},
        common,
    )


# ── Per-task analysis ──────────────────────────────────────────────────────


def analyse_task(task: str) -> dict:
    """Run the full battery on one task, gracefully handling missing methods."""
    available: dict[str, Path] = {}
    for approach in APPROACHES:
        path = find_gen_avg(approach, task)
        if path is not None:
            available[LABELS[approach]] = path

    result: dict = {
        "task": task,
        "available_methods": list(available.keys()),
        "missing_methods": [LABELS[a] for a in APPROACHES if LABELS[a] not in available],
        "skipped": False,
        "per_condition": [],
        "friedman": None,
        "friedman_per_seed": [],
        "pairwise": [],
        "pairwise_holm_t": [],
        "pairwise_holm_w": [],
        # Dispersion analysis (Brown-Forsythe omnibus + variance-ratio CIs)
        "bf_friedman": None,
        "variance_ratios": [],
        "variance_ratio_holm": [],
    }

    if len(available) < 2:
        result["skipped"] = True
        return result

    # Load + align per-subject vectors across the available methods.
    accs_dicts = {label: load_per_subject(path) for label, path in available.items()}
    aligned, common_subjects = align_three(accs_dicts)
    result["n_subjects"] = len(common_subjects)

    # ── Per-condition BCa CI on the mean ───────────────────────────────────
    for i, (label, vec) in enumerate(aligned.items()):
        lo, hi = bca_bootstrap_ci(vec, seed=BOOTSTRAP_SEED + i, n_boot=BOOTSTRAP_N)
        result["per_condition"].append(
            {
                "method": label,
                "n": len(vec),
                "mean": float(vec.mean()),
                "std": float(vec.std(ddof=1)) if len(vec) > 1 else float("nan"),
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )

    # ── Friedman omnibus (requires k >= 3) ─────────────────────────────────
    if len(aligned) >= 3:
        result["friedman"] = run_friedman(aligned)

        # Per-seed robustness Friedman: redo the omnibus on each seed's
        # individual JSON (not the seed-average). Tells us whether the
        # omnibus conclusion is stable across seeds.
        per_seed_files: dict[str, dict[int, Path]] = {
            LABELS[approach]: find_per_seed_jsons(approach, task)
            for approach in APPROACHES
            if LABELS[approach] in aligned
        }
        common_seeds = sorted(set.intersection(*[set(d.keys()) for d in per_seed_files.values()]))
        for seed in common_seeds:
            seed_dicts = {
                label: load_per_subject(per_seed_files[label][seed]) for label in aligned.keys()
            }
            seed_aligned, _ = align_three(seed_dicts)
            seed_result = run_friedman(seed_aligned)
            result["friedman_per_seed"].append((seed, seed_result))
    else:
        # k=2 fallback: report Wilcoxon as the primary test for this task.
        # (Will be picked up by the pairwise section below.)
        pass

    # ── Pairwise post-hoc Wilcoxon (gated on omnibus when k>=3) ────────────
    methods_sorted = list(aligned.keys())
    pairs: list[tuple[str, str]] = []
    for i, m1 in enumerate(methods_sorted):
        for m2 in methods_sorted[i + 1 :]:
            pairs.append((m1, m2))

    omnibus_significant = result["friedman"] is not None and result["friedman"].pval < OMNIBUS_ALPHA
    # When k=2, no omnibus exists — report the single pairwise Wilcoxon directly.
    # When k>=3, only run / report post-hoc if Friedman rejected H0.
    run_pairs = (len(aligned) == 2) or omnibus_significant

    if run_pairs:
        for idx, (m1, m2) in enumerate(pairs):
            paired = run_paired(
                a_label=m1,
                a_vals=aligned[m1],
                b_label=m2,
                b_vals=aligned[m2],
                task=task,
                seed=BOOTSTRAP_SEED + 1000 + idx,
            )
            result["pairwise"].append(paired)

        result["pairwise_holm_t"] = holm_bonferroni([p.t_pval for p in result["pairwise"]])
        result["pairwise_holm_w"] = holm_bonferroni([p.w_pval for p in result["pairwise"]])

    # ── Dispersion analysis ────────────────────────────────────────────────
    # Brown-Forsythe-via-Friedman omnibus (k>=3 only) on absolute deviations
    # from the per-method median. Tests "do the spreads differ at all?".
    if len(aligned) >= 3:
        result["bf_friedman"] = run_brown_forsythe_friedman(aligned)

    # Pairwise BCa CIs on the variance ratio Var(a)/Var(b). Reported for all
    # pairs regardless of the Brown-Forsythe outcome — the CI is informative
    # as an effect size; significance claims still defer to the omnibus.
    # Variance ratio < 1 means method `a` has smaller subject-level spread
    # (more consistent across subjects) than method `b`.
    var_pvals_proxy: list[float] = []
    for idx, (m1, m2) in enumerate(pairs):
        ratio, lo, hi = bca_variance_ratio_ci(
            aligned[m1],
            aligned[m2],
            seed=BOOTSTRAP_SEED + 2000 + idx,
            n_boot=BOOTSTRAP_N,
        )
        # Use the BCa CI as a 2-sided test: "1.0 inside CI" → not significant.
        # Compute a proxy p-value via the fraction-below-1 in the bootstrap
        # ratio distribution to enable Holm correction across pairs.
        # Cheap re-derivation: distance of 1.0 from the centre, normalised by
        # CI half-width, fed through a normal-approx 2-sided p. This is an
        # approximation, not a primary test — the CI itself is the headline.
        if np.isnan(ratio) or np.isnan(lo) or np.isnan(hi):
            approx_p = float("nan")
        else:
            half_width = max((hi - lo) / 2.0, 1e-12)
            z = abs(ratio - 1.0) / (half_width / 1.96)  # 1.96 = z_{0.975}
            from scipy import stats as _stats

            approx_p = float(2 * (1 - _stats.norm.cdf(z)))
        var_pvals_proxy.append(approx_p if not np.isnan(approx_p) else 1.0)
        result["variance_ratios"].append(
            {
                "method_a": m1,
                "method_b": m2,
                "var_a": float(np.var(aligned[m1], ddof=1)),
                "var_b": float(np.var(aligned[m2], ddof=1)),
                "ratio": ratio,
                "ci_lo": lo,
                "ci_hi": hi,
                "approx_p": approx_p,
            }
        )

    result["variance_ratio_holm"] = holm_bonferroni(var_pvals_proxy) if var_pvals_proxy else []

    return result


# ── Markdown emission ──────────────────────────────────────────────────────


def render_markdown(task_results: list[dict], cross_task_holm_friedman: list[float]) -> str:
    L: list[str] = []
    L.append("# Statistical Tests — Generalization")
    L.append("")
    L.append("Companion to `docs/statistical_tests.md`, applied to the FACED")
    L.append("stimulus-generalization runs (`main-results/*_generalization/*_gen_avg.json`,")
    L.append("produced by `src.inference.average_gen_seeds`). The methodology differs")
    L.append("from the main document because the generalization question is omnibus:")
    L.append('*"do the three methods differ at all under held-out stimuli?"* — not')
    L.append('*"is JADE better than each baseline?"*')
    L.append("")
    L.append("Protocol per task:")
    L.append("")
    L.append("1. **Friedman omnibus** across LP, SFT, JADE — primary test on the *means*.")
    L.append("2. **Per-seed Friedman** on each gen seed's individual JSON — robustness check.")
    L.append("3. **Pairwise Wilcoxon signed-rank** on all method pairs, *only if* the")
    L.append("   mean omnibus is significant (α = 0.05), with Holm-Bonferroni across the")
    L.append("   pairs of that task.")
    L.append("4. **BCa bootstrap CIs** on the per-condition mean — identical helper")
    L.append("   to `statistical_tests.py`, so numbers are directly comparable.")
    L.append("5. **Dispersion analysis**: Brown-Forsythe-via-Friedman omnibus on the")
    L.append("   subject-level absolute deviations + pairwise BCa CIs on the")
    L.append("   variance ratio `Var(a)/Var(b)`. Tests whether the methods differ in")
    L.append("   per-subject consistency, independently of mean accuracy.")
    L.append("")
    L.append("Family-wise correction across **tasks** (n=2) is applied to the omnibus")
    L.append("p-values themselves, since each task is one omnibus test in the family.")
    L.append("")

    # ── §0 brief definitions ───────────────────────────────────────────────
    L.append("## 0. Definitions")
    L.append("")
    L.append("**Friedman test.** Non-parametric repeated-measures ANOVA. Each subject")
    L.append("is a block; the k methods are conditions; ranks are assigned per subject")
    L.append("(rank 1 = lowest accuracy for that subject). Under H₀ the average rank")
    L.append("of each method equals (k+1)/2. The test statistic is")
    L.append("`χ² = 12 / (n·k·(k+1)) · Σ R_j² − 3·n·(k+1)`, distributed χ²(k−1) under H₀.")
    L.append("")
    L.append("**Kendall's W.** Effect size for Friedman, `W = χ² / (n·(k−1))`. Bounded")
    L.append("in [0, 1]: 0 = no agreement among subjects on method ranking, 1 = perfect")
    L.append("agreement (all subjects rank the methods identically).")
    L.append("")
    L.append("**Per-seed Friedman.** The seed-averaged accuracies smooth over noise")
    L.append("between the two gen seeds. As a robustness check we rerun Friedman on")
    L.append("each seed's vectors individually; if both per-seed tests agree with the")
    L.append("seed-averaged test, the conclusion is not a smoothing artefact.")
    L.append("")
    L.append("**Pairwise post-hoc Wilcoxon.** Identical to the §2 paired tests in")
    L.append("`statistical_tests.md`. Only computed when Friedman rejects H₀, to avoid")
    L.append("uncontrolled multiple-comparison fishing.")
    L.append("")
    L.append("**Brown-Forsythe-via-Friedman dispersion omnibus.** For each method `m`,")
    L.append("each subject's accuracy `a_i^(m)` is replaced by its absolute deviation")
    L.append("from the method's median: `d_i^(m) = |a_i^(m) − median(a^(m))|`. Friedman")
    L.append("is then applied to the `d` vectors. Tests `H₀`: median absolute deviation")
    L.append("is equal across methods. The median centring (not mean) makes the test")
    L.append("robust to extreme subjects (Brown-Forsythe variant of Levene's test); the")
    L.append("Friedman wrapper preserves the within-subject pairing.")
    L.append("")
    L.append("**Variance-ratio BCa CI.** Paired bootstrap CI on `Var(method_a) /")
    L.append("Var(method_b)`, with subject indices resampled with replacement so the")
    L.append("pairing is preserved. BCa correction is appropriate because the ratio")
    L.append("distribution is bounded below at 0 and asymmetric. A CI excluding 1.0")
    L.append("means the variances differ significantly at the corresponding level.")
    L.append("Variance ratio < 1 means method `a` has tighter subject-level spread")
    L.append("(more consistent across subjects) than method `b`. The CI is reported")
    L.append("regardless of the dispersion-omnibus outcome — it is an effect-size")
    L.append("measure; significance claims still defer to the omnibus.")
    L.append("")

    # ── Per-task sections ──────────────────────────────────────────────────
    for tr_idx, tr in enumerate(task_results):
        task = tr["task"]
        L.append(f"## Task: {task}")
        L.append("")

        if tr["skipped"]:
            L.append(
                f"Skipped — only {len(tr['available_methods'])} method(s) available "
                f"({', '.join(tr['available_methods']) if tr['available_methods'] else 'none'}). "
                f"Missing: {', '.join(tr['missing_methods'])}."
            )
            L.append("")
            continue

        L.append(
            f"Methods available: **{', '.join(tr['available_methods'])}** "
            f"(N = {tr['n_subjects']} subjects)."
        )
        if tr["missing_methods"]:
            L.append("")
            L.append(
                f"Missing: **{', '.join(tr['missing_methods'])}** — re-run this script "
                "once the corresponding `_gen_avg.json` is available."
            )
        L.append("")

        # Per-condition table
        L.append("### Per-condition BCa CI on the mean accuracy")
        L.append("")
        L.append("| Method | N | Mean | Std | 95 % CI on the mean (BCa) |")
        L.append("|--------|---|------|-----|---------------------------|")
        for r in tr["per_condition"]:
            L.append(
                f"| {r['method']:<6} | {r['n']:>3} | "
                f"{r['mean'] * 100:>6.2f} %  | {r['std'] * 100:>5.2f} %    | "
                f"{r['ci_lo'] * 100:>5.2f} % – {r['ci_hi'] * 100:>5.2f} %         |"
            )
        L.append("")

        # Friedman block
        fr: FriedmanResult | None = tr["friedman"]
        if fr is not None:
            adj_p = cross_task_holm_friedman[tr_idx]
            L.append("### Friedman omnibus test")
            L.append("")
            L.append(
                f"- χ²({fr.df}) = **{fr.stat:.3f}**, p = **{fmt_p(fr.pval)}** "
                f"(Holm across tasks, n={len(task_results)}: p = {fmt_p(adj_p)})"
            )
            L.append(f"- Kendall's W = **{fr.kendall_w:.4f}** (effect size in [0, 1])")
            L.append("- Mean ranks (higher rank = better method on a given subject):")
            for lbl in fr.labels:
                L.append(f"    - {lbl}: {fr.mean_ranks[lbl]:.3f}")
            L.append("")

            verdict = (
                "**Reject H₀**: at least one method differs from the others."
                if fr.pval < OMNIBUS_ALPHA
                else "**Fail to reject H₀**: no significant difference among the three methods."
            )
            L.append(verdict)
            L.append("")

            # Per-seed robustness
            if tr["friedman_per_seed"]:
                L.append("### Per-seed Friedman (robustness check)")
                L.append("")
                L.append("| Seed | χ² | df | p (raw) | Kendall's W | Verdict |")
                L.append("|------|-----|----|---------|-------------|---------|")
                for seed, sr in tr["friedman_per_seed"]:
                    verdict = "reject" if sr.pval < OMNIBUS_ALPHA else "fail to reject"
                    L.append(
                        f"| {seed} | {sr.stat:.3f} | {sr.df} | {fmt_p(sr.pval)} | "
                        f"{sr.kendall_w:.4f} | {verdict} |"
                    )
                L.append("")
        else:
            L.append("### Pairwise Wilcoxon (k = 2 fallback)")
            L.append("")
            L.append(
                "Only two methods available — Friedman not applicable. The pairwise"
                " Wilcoxon below is the primary test."
            )
            L.append("")

        # Pairwise post-hoc (gated on omnibus or k=2)
        if tr["pairwise"]:
            if fr is not None:
                L.append("### Pairwise post-hoc Wilcoxon (gated on omnibus)")
                L.append("")
                L.append(
                    f"Friedman significant — running pairwise Wilcoxon on all "
                    f"{len(tr['pairwise'])} pair(s), with Holm-Bonferroni across the family."
                )
                L.append("")
            L.append(
                "| Comparison | Mean Δ (pp) | Cohen's d | Wins / Losses / Ties | "
                "Wilcoxon W | p (Wilcoxon) | Holm p (W) | t | p (t-test) | Holm p (t) |"
            )
            L.append(
                "|------------|-------------|-----------|----------------------|"
                "------------|--------------|------------|---|------------|------------|"
            )
            for p, ph_t, ph_w in zip(tr["pairwise"], tr["pairwise_holm_t"], tr["pairwise_holm_w"]):
                L.append(
                    f"| {p.label_a} vs {p.label_b} | "
                    f"{p.mean_diff * 100:+.2f} | {p.cohen_d:.3f} | "
                    f"{p.n_wins} / {p.n_losses} / {p.n_ties} | "
                    f"{p.w_stat:.0f} | {fmt_p(p.w_pval)} | {fmt_p(ph_w)} | "
                    f"{p.t_stat:+.2f} | {fmt_p(p.t_pval)} | {fmt_p(ph_t)} |"
                )
            L.append("")
        elif fr is not None:
            L.append("### Pairwise post-hoc")
            L.append("")
            L.append(
                "Omnibus not significant (Friedman p ≥ 0.05) — "
                "**post-hoc tests not run** to avoid uncontrolled multiple-comparison fishing."
            )
            L.append("")

        # ── Dispersion analysis ────────────────────────────────────────────
        bf: FriedmanResult | None = tr["bf_friedman"]
        if bf is not None:
            L.append("### Dispersion omnibus (Brown-Forsythe via Friedman)")
            L.append("")
            L.append(f"- χ²({bf.df}) = **{bf.stat:.3f}**, p = **{fmt_p(bf.pval)}**")
            L.append(f"- Kendall's W = **{bf.kendall_w:.4f}**")
            L.append("- Mean ranks of |a − median(a)| per method:")
            for lbl in bf.labels:
                L.append(f"    - {lbl}: {bf.mean_ranks[lbl]:.3f}")
            L.append("")
            bf_verdict = (
                "**Reject H₀**: at least one method differs in dispersion."
                if bf.pval < OMNIBUS_ALPHA
                else "**Fail to reject H₀**: no significant difference in dispersion."
            )
            L.append(bf_verdict)
            L.append("")

        if tr["variance_ratios"]:
            L.append("### Pairwise variance ratios (BCa CIs)")
            L.append("")
            L.append(
                "Variance ratio < 1 ⇒ method `a` has tighter subject-level spread"
                " than method `b`. CI excluding 1.0 ⇒ the difference is significant"
                " at the corresponding level. The Holm-adjusted p column treats the"
                " variance-ratio CIs as a family of tests across pairs of this task."
            )
            L.append("")
            L.append(
                "| Comparison (a vs b) | Var(a) | Var(b) | Ratio Var(a)/Var(b) | "
                "95 % BCa CI | approx p | Holm p |"
            )
            L.append(
                "|----------------------|--------|--------|---------------------|"
                "-------------|----------|--------|"
            )
            for vr, ph in zip(tr["variance_ratios"], tr["variance_ratio_holm"]):
                L.append(
                    f"| {vr['method_a']} vs {vr['method_b']} | "
                    f"{vr['var_a']:.5f} | {vr['var_b']:.5f} | "
                    f"**{vr['ratio']:.3f}** | "
                    f"[{vr['ci_lo']:.3f}, {vr['ci_hi']:.3f}] | "
                    f"{fmt_p(vr['approx_p'])} | {fmt_p(ph)} |"
                )
            L.append("")

    return "\n".join(L) + "\n"


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    task_results = [analyse_task(t) for t in TASKS]

    # Holm-Bonferroni across the omnibus p-values themselves (family of tasks).
    omnibus_pvals = [
        tr["friedman"].pval if tr["friedman"] is not None else 1.0 for tr in task_results
    ]
    cross_task_holm = holm_bonferroni(omnibus_pvals)

    md = render_markdown(task_results, cross_task_holm)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(md)
    print(f"Wrote {OUT_PATH.relative_to(PROJECT_ROOT)}")

    # Brief stdout summary (means + dispersion)
    for tr in task_results:
        if tr["skipped"]:
            print(f"  {tr['task']}: skipped (methods available: {tr['available_methods']})")
            continue
        if tr["friedman"] is not None:
            fr: FriedmanResult = tr["friedman"]
            print(
                f"  {tr['task']}: Friedman χ²({fr.df}) = {fr.stat:.3f}, "
                f"p = {fmt_p(fr.pval)}, W = {fr.kendall_w:.4f}"
            )
        else:
            print(
                f"  {tr['task']}: only {len(tr['available_methods'])} methods — Wilcoxon fallback"
            )
        bf = tr.get("bf_friedman")
        if bf is not None:
            print(
                f"    └ dispersion (Brown-Forsythe via Friedman): χ²({bf.df}) = {bf.stat:.3f}, "
                f"p = {fmt_p(bf.pval)}, W = {bf.kendall_w:.4f}"
            )
        for vr in tr.get("variance_ratios", []):
            ci_str = f"[{vr['ci_lo']:.3f}, {vr['ci_hi']:.3f}]"
            print(
                f"    └ Var({vr['method_a']})/Var({vr['method_b']}) = {vr['ratio']:.3f}  "
                f"95% BCa CI {ci_str}"
            )


if __name__ == "__main__":
    main()
