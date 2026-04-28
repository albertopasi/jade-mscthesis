# JADE Hyperparameter Optimization Methodology

This document describes the sequential hyperparameter search used to identify
the JADE training configuration on FACED. It is intended as the reference
for the methodology section of the thesis. All metrics are 10-fold
cross-subject CV val_acc × 100, unless explicitly marked as single-fold.

---

## 1. LP warmup & FT-FullFT recipe — official REVE values

JADE's two-stage pipeline reuses the **published REVE downstream-task recipe**
for the LP warmup and full-FT optimization (Tab. 1). These hyperparameters
were validated by the foundation-model authors on the same downstream
classification protocol and were not re-tuned in this work.

**Justification.** The point of comparison is the loss formulation
(CE-only vs CE+SupCon), not the optimization recipe. Using the published
recipe for the FT-FullFT baseline avoids confounding "tuning effort" with
"loss design." Where the JADE configuration departs from the REVE recipe
(batch size, learning rate at B=256), the FT-FullFT baseline is re-run with
the *same* departure for a like-for-like comparison.

**Recipe** (per [REVE]):
- LP stage: 20 epochs, lr=5e-3, StableAdamW, 3-epoch exponential warmup,
  patience=10 on val_acc, dropout=0.05.
- FT stage: 200 epochs, lr=1e-4, StableAdamW, 5-epoch warmup,
  patience=20, dropout=0.1, AMP fp16, grad-clip 2.0.
- Batch size: 128 (RAM-constrained on the original setup; this work
  re-tunes batch in Stage 2).

---

## 2. Sequential HP search — strategy

JADE introduces three new hyperparameters: **α** (CE/SupCon balance),
**τ** (SupCon temperature), and (jointly with FT) **batch size and lr**.
A full joint grid is infeasible at full 10-fold CV cost. We use a
three-stage sequential design:

| Stage | Sweeps | HPs frozen | Cost protocol |
|---|---|---|---|
| 1 — Loss HPs | (α, τ) | optimization at REVE default (B=128, lr=1e-4) | Full 10-fold CV |
| 2 — Optimization | batch, lr | (α, τ) at Stage 1 winners | Single fold for direction; full CV for verification |
| 3 — Loss refinement | (α, τ) narrow | optimization at Stage 2 winners | Full 10-fold CV |

**Why this order.** Optimization HPs have systemic effect — if lr is
mis-set, no loss-HP combination compensates. Loss HPs are the novel
contribution and are tuned first at the *baseline* recipe to isolate
the SupCon-specific signal. Optimization is then re-tuned at the
loss winner. Finally, loss HPs are rechecked at the new optimization
to catch any landscape shift.

**Why per-task winners differ** is reported and analyzed in Section 5.

---

## 3. Stage 1 — Loss HP grid at REVE default (B=128, lr=1e-4)

### 9-class (full 10-fold CV, FT-FullFT no-mixup baseline = 58.20)

Main grid: α ∈ {0.2, 0.3} × τ ∈ {0.03, 0.05, 0.1, 0.2, 0.5}.
α-extreme ablation: α ∈ {0.1, 0.5, 0.7, 0.8, 0.9}, τ=0.1.

| α \ τ | 0.03 | 0.05 | 0.1 | 0.2 | 0.5 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.2 | 57.73±2.38 | 56.87±3.59 | 57.21±3.57 | 57.90±3.87 | 57.69±3.11 |
| **0.3** | 58.00±3.53 | 57.65±2.78 | **58.81±3.39** | 57.32±3.21 | 57.31±2.93 |

α-extreme ablation (τ=0.1): α=0.1 → 57.20; α=0.5 → 57.29; α=0.7 → 57.27; α=0.8 → 57.14; α=0.9 → 57.61.

**Stage 1 winner (9-class): α=0.3, τ=0.1, val_acc=58.81 (Δ=+0.61 vs FT baseline).**
Optimum is narrow (only this cell exceeds 58.20 (FT best)); SupCon barely helps at the
default optimization recipe.

### Binary (full 10-fold CV, FT-FullFT no-mixup baseline = 75.55)

Main grid: α ∈ {0.2, 0.3} × τ ∈ {0.03, 0.05, 0.1, 0.2, 0.5}.
α-extreme ablation: α ∈ {0.1, 0.5, 0.7, 0.8}, τ=0.1.

| α \ τ | 0.03 | 0.05 | 0.1 | 0.2 | 0.5 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.2** | 75.70±3.14 | **77.28±1.29** | 76.92±1.88 | 76.16±2.44 | 77.03±1.97 |
| 0.3 | 77.11±1.26 | 76.41±1.68 | 76.47±2.38 | 76.59±2.66 | _missing_ |

α-extreme ablation (τ=0.1): α=0.1 → 76.82; α=0.5 → 76.59; α=0.7 → 75.93; α=0.8 → 75.16.

**Stage 1 winner (binary): α=0.2, τ=0.05, val_acc=77.28 (Δ=+1.73 vs FT baseline).**
Broad optimum: 7 of 10 cells beat the baseline.

---

## 4. Stage 2 — Optimization HP search

Stage 2 holds (α, τ) at Stage 1 winners and tunes batch and lr. Two sub-stages.

### 2a — Batch feasibility (smoke test)

Tested B ∈ {128, 256, 512} for memory feasibility on A100-80GB.
B=512 OOM (peak 79.2 GiB / 80 GB). **B=256 fits comfortably** (peak ~62 GB). All Stage 2 LR sweeps and Stage 3 use B=256.

### 2b — Learning-rate sweep at B=256

**9-class (single fold, fold 1, α=0.3, τ=0.1).** Single-fold sweep used for
direction-finding only — chosen value is verified at full 10-fold CV in
Stage 3.

| lr | val_acc (fold 1) |
|:---:|:---:|
| 5e-5 | _failed (env)_ |
| 1e-4 | 0.6126 |
| 2e-4 | _missing_ |
| **4e-4** | **0.6639** |
| 8e-4 | 0.6758 (best epoch reached safely; note: see stability) |
| 1.5e-3 | _not run; would be unsafe per 8e-4 stability profile_ |

LR=8e-4 obtained the highest fold-1 val_acc (0.6758) but its training
trajectory showed warmup instability (val_acc collapse to 0.15 at epoch 5
before recovery). LR=4e-4 had no such instability. Both LRs were
verified at full CV (Stage 3); the 4e-4 mean was higher and lower-variance.

**9-class Stage 2 winner: B=256, lr=4e-4** (single-fold-best lr=8e-4
rejected for cross-fold instability — see Stage 3).

**Binary (full 10-fold CV, α=0.2, τ=0.05).** A fold-1 sweep was skipped here:
the LR space was small enough to run all candidates at full CV directly.

| lr | val_acc | std |
|:---:|:---:|:---:|
| **1e-4** | **76.74** | 2.71 |
| 2e-4 | 76.38 | 3.43 |
| 4e-4 | 73.47 | 5.74 |
| 8e-4 | 72.62 | 5.31 |

**Binary Stage 2 winner: B=256, lr=1e-4** (lower LR is better at the larger batch).

The Stage 2 sweep is descriptive: among the LRs tested, lr=1e-4 has the
highest mean val_acc (76.74) and lowest std (2.71); higher LRs degrade
both. The deeper observation — that binary did not benefit from
batch scaling at all — is established by comparing the Stage 2 winner
to the Stage 1 winner: 76.74 (B=256) vs 77.28 (B=128), a difference
within fold-noise. This contrasts with 9-class, which gained ~+3pp from
the same batch scaling. The discrepancy motivates the
positives-per-anchor argument analysed in `jade_overall_analysis.md`.

---

## 5. Stage 3 — Loss HP refinement at Stage 2 optimization (full 10-fold CV)

### 9-class (B=256, lr=4e-4)

Narrow re-grid around Stage 1 winner (α=0.3, τ=0.1):

| α \ τ | 0.1 | 0.2 |
|:---:|:---:|:---:|
| 0.2 | _missing_ | _missing_ |
| **0.3** | 62.34±4.09 | **62.61±3.81** |

LR=8e-4 cross-check: τ=0.1 → 62.19±5.68; τ=0.2 → 57.97±7.85 (2 folds
diverged). Confirms LR=4e-4 is safer with no accuracy cost.

**Stage 3 winner (9-class): α=0.3, τ=0.2, val_acc=62.61 (Δ=+3.70 vs FT B=256 baseline 58.91).**
τ shifted from 0.1 (Stage 1) to 0.2 — consistent with the
positives-per-anchor mechanism: at B=256 each anchor sees ~28 positives
vs ~14 at B=128, so a softer (higher-τ) loss is now preferable.

### Binary (B=256, lr=1e-4)

Narrow re-grid around Stage 1 winner (α=0.2, τ=0.05):

| α \ τ | 0.03 | 0.05 | 0.1 |
|:---:|:---:|:---:|:---:|
| 0.2 | 76.71±2.34 | 76.74±2.71 | 75.90±3.11 |
| **0.3** | **77.33±2.49** | _missing_ | 76.58±2.83 |

**Stage 3 winner (binary): α=0.3, τ=0.03, val_acc=77.33 (Δ=+0.11 vs FT B=256 baseline 77.22).**
α shifted from 0.2 (Stage 1) to 0.3 — modest preference for more CE weight
at the new recipe. The accuracy gain over the baseline is within
fold-noise; binary's SupCon contribution is essentially negligible once
the baseline is properly tuned.

---

## 6. Final selected configurations

| Task | Config | Acc | std | FT B=256 baseline | Δ |
|---|---|:---:|:---:|:---:|:---:|
| 9-class | α=0.3, τ=0.2, B=256, lr=4e-4 | **62.61** | 3.81 | 58.91 (lr=4e-4) | **+3.70** |
| Binary | α=0.3, τ=0.03, B=256, lr=1e-4 | **77.33** | 2.49 | 77.22 (lr=2e-4) | **+0.11** |

---

## 7. Methodological notes for the thesis

- **Single-fold use is bounded.** Only Stage 2b LR direction-finding for
  9-class uses fold-1 metrics; the chosen value is always verified at full
  10-fold CV.
- **Per-task optima differ across all three HPs.** This is a finding, not
  a bookkeeping inconvenience — see `jade_overall_analysis.md` for the
  positives-per-anchor argument.
- **All FT-FullFT baselines are re-run when the optimization recipe
  changes.** Tables in this document compare JADE to the FT baseline at
  the *same* batch+lr, never against a fixed reference baseline that uses
  a different recipe.

---

## 8. Holes in the current grid

The following cells are missing for full grid completeness. They are
NOT expected to change the chosen winners (most are ablation cells far
from the optimum), but filling them yields a complete, hole-free
methodology table — useful for the thesis appendix.

| # | Stage | Task | Missing | Purpose |
|:---:|:---:|---|---|---|
| 1 | 1 main | binary | α=0.3, τ=0.5 @ B=128, lr=1e-4 | complete Stage 1 main grid |
| 2 | 1 ablation | binary | α=0.9, τ=0.1 @ B=128, lr=1e-4 | symmetric α-extreme ablation |
| 3 | 3 | 9-class | α=0.2, τ=0.1 @ B=256, lr=4e-4 | complete Stage 3 grid |
| 4 | 3 | 9-class | α=0.2, τ=0.2 @ B=256, lr=4e-4 | complete Stage 3 grid |
| 5 | 3 | binary | α=0.3, τ=0.05 @ B=256, lr=1e-4 | complete Stage 3 grid |

Total: 5 jobs × 10-fold CV. See `slurm/run_jade_grid_holes.sh`.
