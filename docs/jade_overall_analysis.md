# JADE — Overall Analysis & Path Forward

Written after the FACED B=256 follow-up sweep and the THU-EP transfer
attempt. Companion to `docs/jade_vs_ft_results.md`, which holds the raw tables.

This document is opinionated. Where a finding is solid, it says so; where the
evidence is ambiguous, it says that too.

---

## TL;DR

| Setting | JADE Δ vs matched FT baseline | Verdict |
|---|:---:|---|
| FACED 9-class | **+3.70pp** | **Solid win.** SupCon-driven, robust to batch/LR scaling. |
| FACED binary | +0.11pp | **Within noise.** Earlier +1.73pp inflated by FT under-tuning. |
| THU-EP 9-class | −0.09pp | **No transfer.** SupCon's FACED gain disappears. |
| THU-EP binary | −1.31pp | **Negative transfer.** JADE underperforms FT. |

**The thesis headline is FACED 9-class.** Everywhere else, JADE is at-best
neutral or actively worse than a CE baseline.

---

## What was tested

### FACED — exhaustive

- **B=128 (α, τ) sweep** for both tasks. Established initial optima:
  binary (α=0.2, τ=0.05), 9-class (α=0.3, τ=0.1).
- **Single-fold LR sweep at B=256** (lr ∈ {5e-5, 1e-4, 2e-4, 4e-4, 8e-4, 1.5e-3}).
  Identified 4e-4 as 9-class optimum, 1e-4 / 2e-4 stable for binary.
- **B=256 (α, τ) × LR sweep** for both tasks (10-fold CV).
  9-class confirmed α=0.3, τ=0.2 winner; binary best at α=0.3, τ=0.03.
- **FT-FullFT B=256 baselines** at LRs matched to JADE configs.
  Critical for clean attribution.

### THU-EP — direct transfer

- JADE 9-class @ FACED-optimal (α=0.3, τ=0.2, B=256, lr=4e-4)
- JADE binary @ FACED-optimal (α=0.3, τ=0.03, B=256, lr=1e-4)
- FT-FullFT baselines at matched batch+LR.

No THU-EP re-sweep yet — the goal was to test transferability first.

---

## What works, and why

### FACED 9-class JADE: +3.70pp

This is the only result that confidently survives proper baseline control.
The gain is consistent across (α, τ) ∈ {(0.3, 0.1), (0.3, 0.2)} at lr=4e-4
(62.34, 62.61) and significantly exceeds FT-FullFT at the same batch+LR (58.91).

**Why it works:**
- 9-class has many decision boundaries packed into the embedding. CE loss only
  imposes a "correct logit" objective; SupCon adds an explicit pull-positives /
  push-negatives signal that organizes the embedding more cleanly.
- At B=256, each anchor sees ~28 positives (vs ~14 at B=128). This is enough
  to get a useful contrastive gradient estimate.
- Mid-range τ (0.1–0.2) softens the separation pressure to match the noisy
  positive set — sharper τ makes contrastive signal too brittle for few
  positives.

**Why batch matters here**: at B=128, 9-class JADE was barely +0.61pp.
Doubling to B=256 multiplied the gain by ~6×. The mechanism is *positives
density per anchor*: more positives → less variance in the SupCon gradient
estimator → cleaner training signal.

---

## What doesn't work, and why

### FACED binary: SupCon contribution is noise-level

The "+1.73pp SupCon win" reported initially was an artifact of comparing
JADE @ B=128, lr=1e-4 to FT-FullFT @ B=128, lr=1e-4 — but **the FT baseline
LR was never actually tuned**. When we ran FT-FullFT at B=256, lr=2e-4
(which was a natural follow-up after observing 9-class needed scaled LR),
FT alone reaches 77.22. JADE B=256 best is 77.33, Δ=+0.11.

**Why no real binary gain:**
- Binary decision is essentially "is this a positive emotion or a negative
  one." A linear separator on REVE features handles this well — the
  embedding doesn't *need* extra structure imposed.
- At B=128, binary already had ~64 positives per anchor; doubling to ~128
  adds no new gradient information — the SupCon estimator was saturated.
- Properly-tuned CE matches the SupCon-augmented variant on this task.

**What binary *does* gain**: stability. Std drops from FT's 1.21 to JADE's
2.49 — wait, that's actually *worse*. So binary doesn't gain accuracy or
stability. SupCon is dead weight here.

### THU-EP: negative transfer

This is the experiment that complicates the thesis story most.

**What happened:**
- 9-class: FACED 9-class FT 58.91 → JADE 62.61 (+3.70). On THU-EP: FT 47.23
  → JADE 47.14 (−0.09).
- Binary: FACED FT 77.22 → JADE 77.33 (+0.11). THU-EP: FT 70.30 → JADE 68.99 (−1.31).

**The B=256 optimization recipe doesn't even transfer for FT alone.** FT B=64
on THU-EP 9-class is 48.28; FT B=256 is 47.23. Bigger batch slightly hurts FT
on this dataset. So it's not just JADE that fails to transfer — the whole
"scale up batch + LR" recipe is FACED-specific.

**Why THU-EP is different:**

1. **Lower absolute accuracy ceiling.** THU-EP 9-class peaks ~48% across all
   methods (vs FACED ~63%). When the model can barely separate classes,
   contrastive pull doesn't help — pulling embeddings of barely-recognizable
   positives toward each other is noise amplification, not regularization.

2. **Different recording protocol.** 30 channels (A1, A2 removed) vs 32; six
   frequency-band averaged decomposition vs broadband; different EEG cap;
   different stimuli durations and lab. The pretrained REVE features are
   likely less discriminative on this domain.

3. **Subject-level noise dominates.** THU-EP std is in the same range as
   FACED (1.2–2.9pp), but the absolute accuracy is lower, so signal-to-noise
   is much worse. Differences within ±2pp are below detection threshold.

4. **Hyperparameter optima may genuinely differ.** FACED-optimal LR=4e-4 on
   FT made 9-class slightly worse (47.23 < 48.28 at B=64 default lr). The
   batch/LR pair that helps FACED hurts THU-EP — possibly because lower
   gradient noise at B=256 makes the model overfit faster on the harder /
   noisier task, where some implicit regularization is needed.

---

## What this means for the thesis

### What you have

1. **A reproducible +3.70pp gain on FACED 9-class** with controlled baselines.
   This is the headline. Frame it as: SupCon helps when the task has many
   classes packed into the embedding *and* enough positives-per-anchor to
   estimate the contrastive gradient cleanly.

2. **A clean negative result on FACED binary.** Once FT is properly tuned,
   SupCon adds nothing. Useful as evidence that SupCon's value is task-dependent.

3. **A negative-transfer finding on THU-EP.** Both tasks. Don't bury this
   — it's a strong, interesting result that deserves its own section.

### What's risky to overclaim

- **Don't claim "JADE works on EEG emotion recognition."** It works on
  FACED 9-class only. The other three settings range from neutral to
  negative.
- **Don't claim "the recipe transfers."** It doesn't. Cross-dataset
  transfer of both architecture-level (B/LR) and loss-level (α, τ) HPs failed.
- **Don't claim a unified mechanism.** The positives-per-anchor story
  partially explains FACED 9-class vs FACED binary, but doesn't predict
  THU-EP failure.

### A defensible thesis framing

> "We investigated joint CE + supervised contrastive (SupCon) fine-tuning of
> a large EEG foundation model on emotion recognition. On FACED 9-class,
> JADE achieves +3.70pp over a matched-batch CE-only baseline (62.61 vs
> 58.91, 10-fold cross-subject CV). The gain disappears on the binary
> variant of the same dataset, where a properly-tuned CE baseline matches
> SupCon. **The gain does not transfer to a second EEG emotion dataset
> (THU-EP)**: JADE underperforms CE-only baselines on both binary and
> 9-class. We attribute the differential effect to two factors: (i) SupCon
> requires a sufficiently dense positives-per-anchor regime to estimate
> useful gradients, which 9-class FACED at B=256 satisfies and binary
> already saturates; (ii) SupCon assumes the underlying representations are
> discriminative enough that contrastive pull aligns within-class samples
> in semantically meaningful directions, which the lower-accuracy THU-EP
> task does not satisfy."

This frames the negative results as findings, not failures. Reviewers will
respect this far more than a paper that hides the THU-EP result.

---

## How to address the "what doesn't work"

### Path A — Make THU-EP work (most ambitious)

Re-sweep on THU-EP. Hypotheses to test, ordered by likely payoff:

1. **THU-EP may want smaller batch, not larger.** FT B=64 (48.28) > FT B=256
   (47.23) on 9-class. SupCon at B=64 with denser τ may help.
   - Test: JADE B=64 lr=1e-4, α ∈ {0.2, 0.3}, τ ∈ {0.05, 0.1, 0.2}, 9-class.
   - Cost: ~6 jobs × 7h ≈ 42 GPU-h.

2. **The α range may need recalibration.** On a noisier task, more CE weight
   (α=0.5 or 0.7) may be needed to prevent SupCon from dominating with bad
   gradient estimates.
   - Test: B=64 or B=128, α ∈ {0.5, 0.7}, τ=0.1, 9-class.

3. **Different `supcon_repr`** (`mean` or `both` instead of `context`). On
   FACED you only used `context`. THU-EP may benefit from a richer
   representation since context attention may not be reliable when accuracy
   is low.

### Path B — Frame as "task-conditional" finding (least ambitious)

Don't try to fix THU-EP. Report it as evidence that SupCon is dataset-dependent
in EEG emotion recognition, and that the conditions for SupCon to help are
narrow. This is publishable and honest.

### Path C — Add interpretability (intermediate, may rescue the story)

Embedding visualizations / linear probe analyses on JADE-trained vs
FT-trained models. If JADE produces more class-separable embeddings *even on
THU-EP* (despite no accuracy gain), that's a meaningful finding —
representation quality is a separate axis from classification accuracy.

- Cheap to do: just run linear probing or t-SNE on existing checkpoints.
- Could add a "JADE trades accuracy for representation quality on hard
  tasks" angle.

---

## Concrete next-step priorities

I'd run them in this order:

1. **Generalization splits (stimulus-held-out) on FACED for both tasks.**
   You already have JADE-best and FT-best configs. `--generalization` runs
   are 3 jobs each. This tells you whether JADE's FACED 9-class gain is
   robust to OOD stimuli — important for the thesis story to hold under any
   distribution shift.
   - 4 jobs total: JADE×{9cl, bin}, FT×{9cl, bin}, all gen.

2. **Embedding analyses (Path C above).** Run linear probing / t-SNE on
   already-saved JADE checkpoints vs FT checkpoints. Cheap, fast, may
   reframe the THU-EP failure as something more nuanced.

3. **One targeted THU-EP sweep (Path A.1).** Smaller batch, modest sweep.
   Don't sink huge compute — if 6-job sweep doesn't move the needle, accept
   the negative-transfer finding and move to writeup.

4. **Writeup.** Don't keep running experiments. The thesis can be written
   on what you have right now if needed.

### What NOT to do

- Don't sweep more (α, τ) on FACED. Already at fold-noise level.
- Don't try B=512 (OOM).
- Don't repeat already-failed configs hoping for different results.
- Don't run THU-EP with the FACED-optimal config a second time — variance
  is real but the central tendency is clear.

---

## Open questions worth flagging in the thesis

These belong in the "Future Work / Limitations" section rather than as
results:

1. **Does denser positive sampling (e.g., per-class balanced batching)
   help SupCon at small effective batch?** Could rescue 9-class results at
   B=128 and possibly THU-EP.
2. **Is there a relationship between pretrained-model-quality on the
   target dataset and SupCon's marginal benefit?** Hypothesis: SupCon helps
   only when the foundation model's features are *almost* good enough for
   the task. If features are too weak (THU-EP) or already saturated
   (FACED binary), SupCon adds nothing.
3. **Would memory-bank / queue-based contrastive (MoCo-style) help on hard
   tasks where positives are sparse?** Standard SupCon pulls from the batch
   only; a queue would decouple positive count from batch size.

---

## Footnote on methodological honesty

The original "+1.73pp SupCon win on FACED binary" was a real measurement
but compared against an unfairly-tuned FT baseline. When the FT baseline
was properly tuned (lr=2e-4 instead of 1e-4), the gain shrank to +0.11.
This is recorded transparently in `jade_vs_ft_results.md` Section 2 and
should be flagged in the thesis: the binary win was confounded, the
9-class win was not.

This kind of update is normal and expected when running careful
ablations. The lesson generalizes: **always re-tune the baseline at the
same conditions as the experimental method**. Most published "wins" in
foundation-model literature look smaller after this correction.
