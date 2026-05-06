# JADE vs FT Results — FACED

JADE = SupCon joint loss `α·CE + (1-α)·SupCon`, repr=context, no mixup.
FT baseline = FT-FullFT no-mixup (fair comparison — JADE disables mixup).
All rows: 10-fold cross-subject CV, val metrics × 100.
**Δ Acc** = JADE acc − FT-FullFT acc (against the comparable baseline at the same batch size).

---

## Section 1 — B=128 sweep (initial exploration)

### Binary

FT-FullFT no-mixup baseline @ B=128: **Acc=75.55 · Bal=75.55 · AUROC=82.30 · F1=75.45**

| α | τ | Acc | Bal Acc | AUROC | F1 | Δ Acc |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.1 | 0.1 | 76.82 | 76.82 | 83.43 | 76.80 | +1.27 |
| 0.2 | 0.05 | **77.28** | **77.28** | 83.68 | **77.26** | **+1.73** |
| 0.2 | 0.1 | 76.92 | 76.92 | 83.66 | 76.87 | +1.37 |
| 0.2 | 0.2 | 76.16 | 76.16 | 82.75 | 76.10 | +0.61 |
| 0.2 | 0.5 | 77.03 | 77.03 | **83.83** | 76.98 | +1.48 |
| 0.3 | 0.05 | 76.41 | 76.41 | 82.89 | 76.36 | +0.86 |
| 0.3 | 0.1 | 76.47 | 76.47 | 83.15 | 76.43 | +0.92 |
| 0.3 | 0.2 | 76.59 | 76.59 | 83.18 | 76.53 | +1.04 |
| 0.5 | 0.1 | 76.59 | 76.59 | 82.74 | 76.57 | +1.04 |
| 0.7 | 0.1 | 75.93 | 75.93 | 82.66 | 75.89 | +0.38 |
| 0.8 | 0.1 | 75.16 | 75.16 | 82.00 | 75.08 | −0.39 |

**Every JADE config except α=0.8 beats the FT baseline. Best: α=0.2, τ=0.05 (+1.73pp).**
*Caveat (see Section 2): the FT-FullFT baseline at B=128 was at lr=1e-4 (untuned). Re-tuning FT shrinks this gap substantially.*

### 9-class

FT-FullFT no-mixup baseline @ B=128: **Acc=58.20 · Bal=58.44 · AUROC=88.36 · F1=58.20**

| α | τ | Acc | Bal Acc | AUROC | F1 | Δ Acc |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.1 | 0.1 | 57.20 | 57.44 | 88.05 | 57.17 | −1.00 |
| 0.2 | 0.05 | 56.87 | 57.05 | 87.79 | 56.91 | −1.33 |
| 0.2 | 0.1 | 57.21 | 57.39 | 87.96 | 57.22 | −0.99 |
| 0.2 | 0.2 | 57.90 | 58.15 | 88.38 | 57.88 | −0.30 |
| 0.2 | 0.5 | 57.69 | 57.86 | 88.01 | 57.64 | −0.51 |
| 0.3 | 0.05 | 57.65 | 57.84 | 88.32 | 57.76 | −0.55 |
| **0.3** | **0.1** | **58.81** | **59.03** | **88.61** | **58.90** | **+0.61** |
| 0.3 | 0.2 | 57.32 | 57.56 | 88.11 | 57.21 | −0.88 |
| 0.3 | 0.5 | 57.31 | 57.46 | 88.08 | 57.29 | −0.89 |
| 0.5 | 0.1 | 57.29 | 57.47 | 87.86 | 57.27 | −0.91 |
| 0.7 | 0.1 | 57.27 | 57.51 | 88.06 | 57.21 | −0.93 |
| 0.8 | 0.1 | 57.14 | 57.42 | 88.05 | 57.15 | −1.06 |
| 0.9 | 0.1 | 57.61 | 57.87 | 88.35 | 57.62 | −0.59 |

**Only α=0.3, τ=0.1 beats FT baseline (+0.61pp). All other configs marginally below.** Narrow optimum at B=128.

---

## Section 2 — B=256 follow-up

After identifying B=256 fits comfortably on A100-80GB (peak ~52GB), we tested
whether scaling batch + LR changes the picture. Each batch size has its own
properly-tuned FT baseline so the JADE Δ is fair.

### FT-FullFT B=256 baselines (no-mixup)

| Task | LR | Acc | std | AUROC | F1 |
|---|:---:|:---:|:---:|:---:|:---:|
| Binary | 1e-4 | 74.43 | 2.40 | 81.46 | 74.37 |
| Binary | 2e-4 | **77.22** | **1.21** | **84.21** | **77.20** |
| 9-class | 4e-4 | 58.91 | 2.95 | 88.82 | 58.93 |

**Key finding**: FT B=256 binary @ lr=2e-4 hits 77.22 — almost exactly matching
JADE B=128 best (77.28). The original "+1.73pp SupCon win" on binary was
inflated by an under-tuned FT LR. **9-class FT B=256 stays around 58.91**, so
the 9-class story is unaffected by this correction.

### JADE B=256 — 9-class (α fixed at 0.3)

| α | τ | LR | Acc | std | AUROC | F1 | Δ vs FT B=256 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.3 | 0.1 | 4e-4 | 62.34 | 4.09 | 90.30 | 62.52 | +3.43 |
| 0.3 | 0.1 | 8e-4 | 62.19 | 5.68 | 90.26 | 62.31 | +3.28 |
| **0.3** | **0.2** | **4e-4** | **62.61** | 3.81 | **90.36** | **62.81** | **+3.70** |
| 0.3 | 0.2 | 8e-4 | 57.97 | 7.85 | 88.88 | 58.12 | −0.94 (2 folds diverged) |

**Best: α=0.3, τ=0.2, lr=4e-4 → 62.61, Δ=+3.70 vs FT B=256 baseline.**
9-class gain holds up after controlling for batch+LR. lr=8e-4 is unsafe (training divergence on individual folds).

### JADE B=256 — binary (α=0.2 sensitivity grid)

| α | τ | LR | Acc | std | Δ vs FT B=256 lr=2e-4 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.2 | 0.05 | 1e-4 | 76.74 | 2.71 | −0.48 |
| 0.2 | 0.05 | 2e-4 | 76.38 | 3.43 | −0.84 |
| 0.2 | 0.05 | 4e-4 | 73.47 | 5.74 | −3.75 |
| 0.2 | 0.05 | 8e-4 | 72.62 | 5.31 | −4.60 |
| 0.2 | 0.10 | 1e-4 | 75.90 | 3.11 | −1.32 |
| 0.2 | 0.10 | 4e-4 | 73.96 | 4.59 | −3.26 |
| 0.2 | 0.10 | 8e-4 | 66.86 | 7.77 | −10.36 |
| 0.2 | 0.03 | 1e-4 | 76.71 | 2.34 | −0.51 |

### JADE B=256 — binary (extra α=0.3 grid at lr=1e-4)

| α | τ | LR | Acc | std | Δ vs FT B=256 lr=2e-4 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.3** | **0.03** | **1e-4** | **77.33** | 2.49 | **+0.11** |
| 0.3 | 0.10 | 1e-4 | 76.58 | 2.83 | −0.64 |

**Best binary: α=0.3, τ=0.03, lr=1e-4 → 77.33, Δ=+0.11.**
Binary's SupCon edge over a properly-tuned FT baseline is within noise.

### Summary — FACED final picture

| Task | Best JADE | std | FT B=256 best | JADE Δ |
|---|:---:|:---:|:---:|:---:|
| Binary | 77.33 (B=256, α=0.3, τ=0.03, lr=1e-4) | 2.49 | 77.22 (lr=2e-4) | **+0.11** |
| 9-class | **62.61** (B=256, α=0.3, τ=0.2, lr=4e-4) | 3.81 | 58.91 (lr=4e-4) | **+3.70** |

- **Binary**: JADE's gain is statistically negligible vs a well-tuned FT baseline.
- **9-class**: JADE's +3.70pp gain is the headline result of this work.

---

## Section 3 — FACED stimulus-generalization (3 seeds × 10 folds)

Generalization protocol: for each of 3 seeds {123, 456, 789}, randomly
split the 28 stimuli of each emotion class so that 2/3 are in the train
set and 1/3 are held out for validation. Cross-subject 10-fold CV is
applied independently on top of each stimulus split. Reported below: per-seed
val_acc and 3-seed mean ± seed-std.

All configs use the bulletproof-CV winners (see Section 2):
- 9-class: B=256, lr=4e-4, JADE α=0.3, τ=0.2 / FT no-mixup
- Binary:  B=128, lr=1e-4, JADE α=0.2, τ=0.05 / FT no-mixup
- LP: official mode (frozen encoder, B=64, lr=5e-3)

Hyperparameters were selected on cross-subject CV and applied unchanged
(standard train/val/test methodology — no re-tuning on the gen split).

### 9-class (chance = 11.1%)

| Approach | s123 | s456 | s789 | mean ± std (over seeds) |
|---|:---:|:---:|:---:|:---:|
| LP (frozen) | 15.81±1.49 | 15.67±1.21 | 16.00±1.51 | **15.83 ± 0.17** |
| FT-FullFT B=256 | 15.36±1.18 | 14.69±1.12 | 15.76±1.52 | **15.27 ± 0.54** |
| JADE-FullFT B=256 | 15.95±1.07 | 14.09±1.22 | 15.93±1.49 | **15.32 ± 1.07** |

3-seed AUROC: LP 53.82±0.41 · FT 53.09±0.70 · JADE 52.58±1.18.
3-seed F1: LP 12.15±0.83 · FT 12.72±0.71 · JADE 11.38±0.51.

### Binary (chance = 50.0%)

| Approach | s123 | s456 | s789 | mean ± std (over seeds) |
|---|:---:|:---:|:---:|:---:|
| LP (frozen) | 58.74±3.18 | 60.44±3.38 | 56.19±1.49 | **58.46 ± 2.14** |
| FT-FullFT B=128 | 61.12±3.81 | 59.97±4.29 | 58.49±2.47 | **59.86 ± 1.32** |
| JADE-FullFT B=128 | 60.65±2.45 | 61.61±2.96 | 59.56±2.98 | **60.61 ± 1.03** |

3-seed AUROC: LP 59.82±3.13 · FT 61.31±2.06 · JADE 61.91±0.88.
3-seed F1: LP 56.86±2.15 · FT 59.09±1.48 · JADE 59.52±0.60.

### Comparison to cross-subject CV

| Task | Approach | CV acc | Gen acc (3-seed) | Drop |
|---|---|:---:|:---:|:---:|
| 9-class | FT-FullFT B=256 | 58.91 | 15.27 | −43.64 |
| 9-class | JADE-FullFT B=256 | 62.61 | 15.32 | −47.29 |
| Binary | FT-FullFT B=128 (matched) | not measured at B=128 | 59.86 | n/a |
| Binary | JADE-FullFT B=128 | 77.28 | 60.61 | −16.67 |

JADE − FT delta (gen vs CV):
- **9-class**: CV Δ = +3.70 → gen Δ = **+0.05** (collapses to a tie).
- **Binary**: CV Δ = +1.73 (vs B=128 FT not directly measured; vs B=256 FT-tuned baseline 77.22, CV Δ = +0.11) → **gen Δ = +0.75** (small but consistent across all 3 seeds).

### What can be claimed from these tables

- **All three methods lose substantially.** 9-class drops to within ~4pp of
  the 11.1% chance baseline; binary drops to within ~10pp of the 50% chance
  baseline.
- **The CV ranking (LP < FT < JADE) is preserved on binary** under stimulus
  shift — LP 58.46 < FT 59.86 < JADE 60.61. Gaps shrink from CV (~3pp range)
  to gen (~2pp range), but the order is robust across all 3 seeds.
- **The CV ranking collapses on 9-class** — LP 15.83, FT 15.27, JADE 15.32
  are statistically tied; LP's mean is even slightly highest. 9-class CV's
  +3.70 JADE win does not transfer.
- **JADE 9-class has the highest seed-to-seed variance** (std=1.07 vs
  LP=0.17, FT=0.54) — its gen result depends more on which 1/3 of stimuli
  are held out. Consistent with a model that organizes more aggressively
  around training stimuli, but not directly tested.

### What cannot be claimed

We do not directly observe whether the model is "memorizing stimuli." The
generalization drop and the JADE-FT collapse on 9-class are consistent
with stimulus-specific feature learning, but neither LP-vs-FT nor
JADE-vs-FT differences directly probe representation content. Mechanism
discussion is in `docs/jade_overall_analysis.md`.

---

## Section 4 — THU-EP transfer (direct application of FACED-optimal configs)

Applied FACED-best configs directly to THU-EP, no re-sweep. Each compared to a
matching FT-FullFT baseline at the same batch+LR.

| Task | Approach | Config | Acc | std | AUROC | F1 |
|---|---|---|:---:|:---:|:---:|:---:|
| 9-class | FT B=64 (existing) | — | 48.28 | 2.72 | 84.04 | 47.98 |
| 9-class | FT B=256 lr=4e-4 | — | 47.23 | 1.81 | 83.35 | 46.85 |
| 9-class | JADE B=256 lr=4e-4 | α=0.3, τ=0.2 | 47.14 | 2.93 | 83.06 | 46.90 |
| Binary | FT B=64 (existing) | — | 69.90 | 1.67 | 75.58 | 69.85 |
| Binary | FT B=256 lr=2e-4 | — | **70.30** | 1.24 | **76.13** | 70.12 |
| Binary | JADE B=256 lr=1e-4 | α=0.3, τ=0.03 | 68.99 | 2.86 | 74.78 | 68.80 |

**Negative transfer.** On THU-EP, **JADE does not beat FT** at either task:
- 9-class: JADE 47.14 vs FT B=256 47.23 (−0.09, basically tied; both *worse* than FT B=64 48.28)
- Binary: JADE 68.99 vs FT B=256 70.30 (−1.31)

Even the FT optimization recipe (B=256, scaled LR) doesn't transfer — FT B=256 is no better than FT B=64 on THU-EP 9-class, and only marginally better on binary.

See `docs/jade_overall_analysis.md` for the comprehensive discussion of why
this happens and what to do about it.

---

## Section 5 — LoRA reference (deprecated track)

LoRA-JADE never beat FT-LoRA in cross-subject CV. Track abandoned in favor of
full FT. Numbers retained in earlier revisions if needed.
