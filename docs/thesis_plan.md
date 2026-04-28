# Thesis Plan — Going Forward

This document is the operative plan after the FACED B=256 sweep, the THU-EP
transfer attempt, and the methodology audit. It supersedes any earlier
informal plan.

---

## TL;DR

**Scope**: FACED-deep-dive thesis with two main axes — JADE-FullFT
hyperparameter optimization and LoRA vs FullFT comparison. Generalization
splits validate JADE's robustness within FACED. THU-EP is a secondary
"cross-dataset transfer" finding (already executed, reported as
limitation).

**Why this scope**: a clean single-dataset thesis with a well-defended HP
methodology, a parameter-efficient extension (LoRA), and a transparent
negative-transfer finding is more defensible than a sprawling multi-dataset
thesis with shallow coverage of each.

---

## Phase 1 — Bulletproof JADE-FullFT HP sweep on FACED

**Status: in progress.**

Goal: produce an HP sweep that cannot be attacked on methodology grounds.
Defines `α`, `τ`, batch size, and LR for both binary and 9-class tasks.

**Three-stage sequential design** (see `docs/jade_hp_methodology.md`):
1. Stage 1 — `(α, τ)` grid at REVE default optimization (B=128, lr=1e-4),
   full 10-fold CV.
2. Stage 2 — batch size + LR at Stage 1 winners. B=256 chosen by memory
   feasibility. LR by single-fold direction-finding (9-class) or full CV
   (binary), then verified at full CV.
3. Stage 3 — narrow `(α, τ)` re-grid at Stage 2 winners, full CV.

**Bulletproofing in flight** (12 jobs, `slurm/run_jade_bulletproof.sh`):
- Stage 1 grid holes filled (binary α=0.3 τ=0.5; α=0.9 τ=0.1).
- Stage 3 plus-shapes around per-task winners filled.
- Stage 3 LR cross-check: 9-class winner at lr=1e-4 confirms no shift.
- Plus the in-flight `slurm/run_lr_holes.sh` (9-class lr=2e-4 cross-check).

**Current best configurations (subject to bulletproof results)**:

| Task | Config | Acc | std | Δ vs FT B=256 |
|---|---|:---:|:---:|:---:|
| 9-class | α=0.3, τ=0.2, B=256, lr=4e-4 | 62.61 | 3.81 | +3.70 |
| Binary | α=0.3, τ=0.03, B=256, lr=1e-4 | 77.33 | 2.49 | +0.11 |

**Done when**: bulletproof script complete, methodology doc tables filled
(no remaining holes).

---

## Phase 2 — Generalization splits on FACED

**Status: in flight (`slurm/run_faced_generalization.sh`).**

Goal: test whether JADE's FACED gain holds under stimulus distribution
shift (2/3 stimuli per class in train, 1/3 held out for val). Validates
the in-distribution result against a stronger generalization criterion.

**Submitted (4 jobs, 3 seeds × 10 folds each)**:
- JADE 9-class @ α=0.3 τ=0.2 B=256 lr=4e-4
- FT-FullFT 9-class @ B=256 lr=4e-4
- JADE binary @ α=0.3 τ=0.03 B=256 lr=1e-4
- FT-FullFT binary @ B=256 lr=2e-4

**Possible outcomes**:
- *9-class JADE Δ preserved* (≥+2pp over FT): strong robustness claim.
- *9-class JADE Δ shrinks but positive*: still usable; SupCon
  partially OOD-robust.
- *9-class JADE Δ becomes negative*: SupCon overfits to in-distribution
  stimulus structure. Important finding; would trigger a discussion of
  what SupCon is actually learning.

**Done when**: cross-seed summaries produced, results added to
`docs/jade_vs_ft_results.md` Section 3.

---

## Phase 3 — LoRA vs FullFT comparison on FACED

**Status: planned, not yet submitted.**

Goal: test whether the parameter-efficient LoRA matches FullFT *at the
properly-tuned recipe* and whether JADE's contribution is preserved.

**Motivation**: existing LoRA runs were performed only at the un-tuned
B=128, lr=1e-4 recipe with a sparse `(α, τ)` grid (only 1 cell for
9-class). The "LoRA never beat FT" conclusion is based on insufficient
evidence — the same un-tuned recipe also made FT-FullFT look weaker than
it is.

**Plan (4 jobs, 10-fold CV each)**:
1. FT-LoRA 9-class @ B=256 lr=4e-4 no-mixup
2. FT-LoRA binary @ B=256 lr=2e-4 no-mixup
3. JADE-LoRA 9-class @ Stage 3 9-class winner (α=0.3, τ=0.2, B=256, lr=4e-4)
4. JADE-LoRA binary @ Stage 3 binary winner (α=0.3, τ=0.03, B=256, lr=1e-4)

This produces a clean 2×2×2 cell array (FullFT vs LoRA × FT vs JADE × task).

**Why only 4 jobs (not a full Stage 3 grid for LoRA)**:
- Reuses FullFT-derived `(α, τ)` winners; no LoRA-specific grid search.
- Establishes whether LoRA is in the same ballpark first; expand to a
  grid only if LoRA matches/beats FullFT.

**Possible findings (any are publishable)**:
- *LoRA matches FullFT*: clean parameter-efficiency win. Strongest
  outcome — practical value for real-world EEG fine-tuning. Triggers a
  proper LoRA Stage 3 grid.
- *LoRA still underperforms*: defensible negative — "LoRA is insufficient
  for full FACED capacity." Honest scope statement.
- *JADE's gain differs across LoRA vs FullFT*: differential mechanistic
  finding. Most interesting if the gain is *preserved* under LoRA
  (suggests SupCon doesn't need extra capacity) or *lost* (suggests it
  does).

**Done when**: 4 jobs complete, results compared at matched-recipe FT
baselines, methodology doc gains a "Section 6 — LoRA comparison" or
similar.

---

## Phase 4 — THU-EP transfer (already executed; report as limitation)

**Status: done. To be written up as limitation.**

Goal: cross-dataset transfer test of FACED-optimal configurations.

**Already-run configs (no further compute planned)**:
- JADE THU-EP 9-class @ FACED-optimal config: 47.14 ± 2.93
- FT-FullFT THU-EP 9-class @ B=256 lr=4e-4: 47.23 ± 1.81
- JADE THU-EP binary @ FACED-optimal config: 68.99 ± 2.86
- FT-FullFT THU-EP binary @ B=256 lr=2e-4: 70.30 ± 1.24

**Outcome**: negative transfer on both tasks. JADE does not beat FT
on THU-EP at the FACED-tuned recipe. The optimization recipe itself
(B=256, scaled LR) also doesn't transfer for FT — FT-B=64 is competitive
or better than FT-B=256 on THU-EP.

**Reporting plan**:
- 1-page subsection in thesis: "Cross-dataset transfer of FACED-tuned
  recipe."
- Acknowledge: SupCon's value depends on (a) sufficient
  positives-per-anchor density for the contrastive estimator, and (b)
  the underlying foundation model producing discriminative-enough
  features that contrastive pull aligns within-class samples in
  semantically-meaningful directions.
- THU-EP appears to violate (b): all methods cap at 47-48% accuracy on
  9-class (vs FACED 58-62%), suggesting REVE's pretrained features are
  weaker on this domain.
- Frame as a *finding* about scope, not a methodological failure.

**Done when**: 1 paragraph in thesis discussion + the existing
`docs/jade_overall_analysis.md` is referenced.

**If time permits — optional Phase 4 extension**:
A targeted THU-EP re-sweep (smaller batch, possibly different α range)
to test whether THU-EP has its own optimum that *would* show a JADE gain.
Cost: ~10-20 GPU-h. Risk: may still fail. Decision: do this only if
Phases 1-3 finish cleanly with time to spare.

---

## Sequencing & expected timeline

Wall-clock at 2-concurrent SLURM limit:

| Phase | Compute | Wall-clock |
|---|:---:|:---:|
| 1 — Bulletproof FACED sweep | 12 jobs × ~6h | ~3 days |
| 2 — Generalization splits | 4 jobs × ~25h | ~3 days (parallel-friendly) |
| 3 — LoRA comparison | 4 jobs × ~6h | ~1 day |
| 4 — Writeup of THU-EP | none | — |

**Phases 1, 2, 3 can run in parallel** within SLURM concurrency limits.
The 12 + 4 + 4 = 20 jobs would total ~140 GPU-h. At 2-concurrent that's
~3.5 wall-clock days, partly overlapping with thesis writing.

---

## What gets reported in the thesis

Roughly per chapter:

1. **Introduction & related work** — REVE foundation model, SupCon, EEG
   emotion recognition.
2. **Method** — JADE architecture (joint CE + SupCon), training pipeline,
   datasets.
3. **Hyperparameter methodology** — `docs/jade_hp_methodology.md` content;
   sequential search design with full Stage 1/2/3 tables.
4. **Main results — FACED FullFT** — JADE +3.70pp on 9-class; +0.11 on
   binary (within noise). Deep discussion of why per-task results differ
   (positives-per-anchor mechanism).
5. **Generalization** — Phase 2 results on stimulus-held-out splits.
6. **Parameter efficiency — LoRA** — Phase 3 results.
7. **Cross-dataset transfer — THU-EP** — Phase 4: brief, honest, framed as
   scope-of-method.
8. **Discussion / limitations / future work** — open questions
   (memory-bank contrastive, alternative SupCon variants, SEED extension).

---

## Decisions explicitly **out of scope**

- **SEED dataset integration**: 1-2 days code + sweep weeks. The LoRA
  axis substitutes for a second-dataset extension and is cheaper.
- **Linear probing analyses**: LP runs were used only as warmup stages
  for FT/JADE; no separate LP results are claimed.
- **Memory-bank contrastive (MoCo-style)**: discussed in future-work
  section. Not implemented.
- **Mixed mixup + SupCon**: SupCon requires clean labels; mixup is
  disabled when JADE is active. Documented but not ablated.
- **τ < 0.03 or τ > 0.5**: outside the standard SupCon range; not swept.
- **α ∈ {0.4, 0.6}**: between tested values; bracketed by adjacent cells.
