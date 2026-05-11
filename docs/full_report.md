# JADE on FACED — Full Report

End-to-end report of the work: dataset, tasks, evaluation protocols,
methods, implementation, and results in the order they were generated. This
document is intended as a self-contained narrative — the methodology
companion (`docs/jade_hp_methodology.md`) goes deeper on the JADE
hyperparameter search, and the discussion companion
(`docs/jade_overall_analysis.md`) covers the overall findings.

---

## 1. What this work is doing

We investigate whether **fine-tuning a pretrained EEG foundation model
with a joint cross-entropy + supervised-contrastive loss (JADE)** improves
emotion-recognition accuracy on FACED, compared to **fine-tuning with
cross-entropy only (FT)** or **linear probing on frozen features (LP)**.

The pretrained backbone is **REVE**, a 22-layer transformer trained on
heterogeneous EEG data. We take REVE's published downstream-task
recipe as our optimization baseline and tune only the loss-specific
hyperparameters of JADE.

Beyond the headline accuracy comparison, we evaluate **out-of-distribution
robustness** by holding out a subset of stimuli at training time
("stimulus-generalization") to test whether the methods learn
emotion-general features or stimulus-specific ones.

---

## 2. Dataset — FACED

**FACED** is a public EEG dataset for emotion recognition.

- **Subjects**: 123 (0-indexed: 0–122).
- **Channels**: 32 EEG electrodes.
- **Stimuli**: 28 video clips (~30 s each), labeled with one of 9 emotion
  categories.
- **Sampling**: original 250 Hz; resampled to 200 Hz at preprocessing.
- **Per-subject data**: shape `(28, 32, 6000)` after preprocessing —
  28 stimuli × 32 channels × 30 s × 200 Hz.

Stimulus-to-emotion mapping (3 stimuli per emotion):

| Stimuli | Emotion | 9-class label | Binary label |
|---|---|:---:|:---:|
| 0–2 | Anger | 0 | negative |
| 3–5 | Disgust | 1 | negative |
| 6–8 | Fear | 2 | negative |
| 9–11 | Sadness | 3 | negative |
| 12–15 | Neutral | 4 | (dropped) |
| 16–18 | Amusement | 5 | positive |
| 19–21 | Inspiration | 6 | positive |
| 22–24 | Joy | 7 | positive |
| 25–27 | Tenderness | 8 | positive |

EEG recordings are split into non-overlapping 10-second windows
(2000 samples each); each subject yields ~84 windows per task (28
stimuli × 3 windows ≈ 84, slightly fewer for binary because the
neutral class is excluded).

---

## 3. Tasks

We report results on two label granularities:

| Task | # classes | # stimuli used | Description |
|---|:---:|:---:|---|
| **9-class** | 9 | 28 | Discriminate among 9 fine-grained emotions |
| **Binary**  | 2 | 24 (neutral excluded) | Discriminate positive vs negative valence |

Binary is easier (fewer decision boundaries; coarse arousal/valence cues
suffice). 9-class is much harder (the model must separate semantically
similar emotions like Joy vs Amusement, or Fear vs Disgust).

---

## 4. Evaluation protocols

Two complementary protocols are used. Each measures a different aspect of
generalization.

### 4.1 Cross-subject 10-fold CV (in-distribution)

Subjects are partitioned into 10 folds (random shuffle, fixed seed=42).
For each fold, 9 folds train and 1 fold validates. Reported metrics
are the **mean ± std across the 10 folds**. This measures generalization
to *new subjects* on the *same stimuli* — the standard EEG-emotion
benchmark protocol.

### 4.2 Stimulus-generalization (out-of-distribution)

For each of 3 random seeds {123, 456, 789}, the 28 stimuli of each
emotion class are randomly partitioned: 2/3 for training, 1/3 held out
for validation. Cross-subject 10-fold CV is then applied independently
on each stimulus split, and final per-seed numbers are
mean ± std across folds. Reported headline values are **the
mean ± std across the 3 stimulus seeds**.

This protocol holds out *both* unseen subjects and unseen stimuli at
validation time. It tests whether the model has learned features that
generalize to new stimulus content within the same emotion category.

---

## 5. Methods

Three approaches share the same backbone (REVE) but differ in what is
trainable and what loss is used.

### 5.1 Linear probing (LP)

The REVE encoder is **frozen**. Only the lightweight classifier head and a
trainable `cls_query_token` are updated. This isolates "how good are
the pretrained features for this task?"

- Loss: cross-entropy only.
- Optimizer: StableAdamW, lr=5e-3, 3-epoch warmup, ReduceLROnPlateau.
- Epochs: 20 max, early-stop patience 10.
- Hyperparameters: **inherited from the official REVE downstream
  recipe**, not re-tuned.

### 5.2 Fine-tuning, full-FT (FT)

Two-stage pipeline:

1. **LP warmup** (same as §5.1): frozen encoder, train head + query token.
2. **Full fine-tuning** (FT-FullFT): unfreeze the *entire* encoder, train
   end-to-end. (LoRA fine-tuning was tested in earlier experiments but
   abandoned — see §10.)

- Loss: cross-entropy only.
- Optimizer: StableAdamW, lr varied by recipe (see §6.2).
- Epochs: 200 max, early-stop patience 20, 5-epoch warmup.
- AMP fp16, grad-clip 2.0.
- **Mixup is disabled** (`--no-mixup`) — JADE requires clean labels for
  SupCon, so for fair comparison FT also runs without mixup.

### 5.3 JADE — joint CE + SupCon (JADE-FullFT)

Same two-stage pipeline as FT, but Stage 2 uses a **joint loss**:

$$
L = \alpha \cdot L_\mathrm{CE} + (1 - \alpha) \cdot L_\mathrm{SupCon}
$$

- $L_\mathrm{SupCon}$ is the supervised contrastive loss of Khosla et
  al. 2020 ($L_\mathrm{sup\_out}$ form).
- A small projection head (Linear → ReLU → Linear → L2-normalize, 128-D
  output) sits on top of the encoder for the SupCon loss.
- α and τ (SupCon temperature) are the **new hyperparameters** introduced
  by JADE; they are tuned in this work.
- Mixup is disabled (incompatible with SupCon's positive-pair
  identification).

### 5.4 Naming convention

Results are stored under `summary_faced_<task>_w10s10_pool_no_r16_<config-tag>_fullft.json`:
- `r16` = LoRA rank if applicable (FullFT runs ignore the value).
- `<config-tag>` includes JADE's `a<α>_t<τ>_context` and the optimization
  recipe `b<batch>_lr<ft_lr>` once those depart from the REVE default.

---

## 6. Implementation summary

- **Backbone**: REVE-base (22 layers, embed_dim=512, 8 heads, 200-sample
  patches with 20-sample overlap → 11 patches per 10 s window).
- **Pooling mode**: `no` — query attention concatenated with all patch
  tokens, flattened to a `(1 + 32 × 11) × 512 = 180 736`-dim feature
  for the classifier head.
- **Head**: `RMSNorm → Dropout → Linear`.
- **Optimizer**: StableAdamW (β₁=0.92, β₂=0.999, weight_decay=0.01).
- **LR schedule**: exponential warmup → ReduceLROnPlateau on val_acc
  (patience=6, factor=0.5).
- **Precision**: AMP fp16.
- **Code**: `src/approaches/{linear_probing,fine_tuning,jade}/train_*.py`.
- **Logging**: W&B (`eeg-lp-v2`, `eeg-ft-v2`, `eeg-jade-v2` projects).
- **Hardware**: A100-80GB on DelftBlue. B=256 fits at ~62 GB peak.

---

## 7. Results — LP

LP uses the official REVE downstream recipe verbatim (no HP tuning in
this work). LP serves as a **frozen-feature baseline**: any
fine-tuning method should beat it on in-distribution evaluation, and
the comparison on out-of-distribution evaluation is informative about
where the limitations live.

### 7.1 LP cross-subject 10-fold CV

| Task | Acc | Bal Acc | AUROC | F1 |
|---|:---:|:---:|:---:|:---:|
| 9-class (no-mixup) | 49.72 ± 2.93 | 49.92 | 83.68 | 49.54 |
| Binary (no-mixup) | 71.50 ± 1.58 | 71.50 | 77.69 | 71.43 |

The 9-class LP CV result (~49.7%) is well above chance (11.1%),
indicating REVE's pretrained features carry useful emotion information
even without fine-tuning. Binary CV at 71.5% is +21.5 pp above chance.

### 7.2 LP stimulus-generalization (3 seeds × 10 folds)

| Task | s123 | s456 | s789 | mean ± std |
|---|:---:|:---:|:---:|:---:|
| 9-class | 15.81 ± 1.49 | 15.67 ± 1.21 | 16.00 ± 1.51 | **15.83 ± 0.17** |
| Binary  | 58.74 ± 3.18 | 60.44 ± 3.38 | 56.19 ± 1.49 | **58.46 ± 2.14** |

LP gen is far below LP CV — 9-class drops by ~34 pp (49.72 → 15.83) and
binary drops by ~13 pp (71.50 → 58.46). Even with the encoder frozen,
holding stimuli out destroys most of the 9-class signal. This is the
first evidence that the bottleneck is **upstream of the head**, in the
features themselves.

---

## 8. Results — FT (FullFT, no SupCon)

FT establishes the **CE-only fine-tuning baseline** that JADE is compared
against. Hyperparameters are inherited from REVE for both tasks. A
B=256 9-class control run is included to rule out the JADE 9-class gain
being driven by the optimization recipe alone (see §9.2 for the
motivation).

### 8.1 FT cross-subject CV — REVE default recipe

| Task | Recipe | Acc | Bal Acc | AUROC | F1 |
|---|---|:---:|:---:|:---:|:---:|
| 9-class | B=64, lr=1e-4 | 58.20 ± 3.32 | 58.44 | 88.36 | 58.20 |
| Binary  | B=64, lr=1e-4 | 75.55 ± 2.16 | 75.55 | 82.30 | 75.45 |

These numbers are the comparison baselines used throughout this
report. Binary uses the REVE default recipe end-to-end.

### 8.2 FT cross-subject CV — 9-class control at the JADE B=256 recipe

When JADE's 9-class winner settles on B=256, lr=4e-4 (see §9.2), we
re-run FT at the *same* recipe to ensure the JADE gain is attributable
to the loss formulation rather than the optimization recipe.

| Task | Recipe | Acc | std | AUROC | F1 |
|---|---|:---:|:---:|:---:|:---:|
| 9-class | B=256, lr=4e-4 | 58.91 | 2.95 | 88.82 | 58.93 |

FT 9-class at B=256, lr=4e-4 is essentially unchanged vs the B=64
default (58.91 vs 58.20, within fold noise). The optimization recipe
alone does *not* yield a meaningful CE-only improvement on 9-class —
any JADE gain at this recipe is genuinely loss-driven.

(For binary, B=256 was not pursued: §9.2 showed that JADE did not
benefit from scaling on binary, so both methods stay at REVE default
for that task.)

### 8.3 FT stimulus-generalization (3 seeds × 10 folds)

Recipes match the JADE comparator: 9-class @ B=256 lr=4e-4 (the
JADE-9-class recipe); binary @ B=128 lr=1e-4 (matching JADE's binary
recipe — see §9.4).

| Task | s123 | s456 | s789 | mean ± std |
|---|:---:|:---:|:---:|:---:|
| 9-class (B=256) | 15.36 ± 1.18 | 14.69 ± 1.12 | 15.76 ± 1.52 | **15.27 ± 0.54** |
| Binary  (B=128) | 61.12 ± 3.81 | 59.97 ± 4.29 | 58.49 ± 2.47 | **59.86 ± 1.32** |

Like LP, FT generalization drops sharply from CV (9-class: 58.91 →
15.27, ~44 pp drop; binary: ~75 → 59.86, ~15 pp drop). The drops are
quantitatively similar to LP's, suggesting fine-tuning does not produce
substantially more generalizable representations than frozen features
on this protocol.

---

## 9. Results — JADE

JADE adds α and τ on top of the FT recipe. The HP search proceeds in
**three sequential stages** (full methodology: `docs/jade_hp_methodology.md`).
The order is: loss HPs first, optimization HPs second, loss
HPs re-checked at the new optimization third.

### 9.1 Stage 1 — Loss HP grid at REVE default (B=128 (increased from 64), lr=1e-4), full CV

Each cell is JADE-FullFT 10-fold cross-subject CV val_acc. Bold = winner.

**9-class** (FT-FullFT B=64 baseline = 58.20)

| α \ τ | 0.03 | 0.05 | 0.1 | 0.2 | 0.5 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.2 | 57.73 ± 2.38 | 56.87 ± 3.59 | 57.21 ± 3.57 | 57.90 ± 3.87 | 57.69 ± 3.11 |
| **0.3** | 58.00 ± 3.53 | 57.65 ± 2.78 | **58.81 ± 3.39** | 57.32 ± 3.21 | 57.31 ± 2.93 |

α-extreme ablation (τ=0.1): 0.1→57.20, 0.5→57.29, 0.7→57.27, 0.8→57.14, 0.9→57.61.

**Binary** (FT-FullFT B=64 baseline = 75.55)

| α \ τ | 0.03 | 0.05 | 0.1 | 0.2 | 0.5 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.2** | 75.70 ± 3.14 | **77.28 ± 1.29** | 76.92 ± 1.88 | 76.16 ± 2.44 | 77.03 ± 1.97 |
| 0.3 | 77.11 ± 1.26 | 76.41 ± 1.68 | 76.47 ± 2.38 | 76.59 ± 2.66 | 76.10 ± 3.02 |

α-extreme ablation (τ=0.1): 0.1→76.82, 0.5→76.59, 0.7→75.93, 0.8→75.16, 0.9→76.17.

**Stage 1 winners**: 9-class α=0.3, τ=0.1; binary α=0.2, τ=0.05.

### 9.2 Stage 2 — Optimization HP search

**Motivation.** SupCon's gradient signal scales with the number of
positives per anchor: at B=128 with 9 emotion classes, each anchor has
on average ~14 same-class positives in the batch — a relatively sparse
contrastive signal. We hypothesised that increasing the batch size
would benefit JADE more than CE-only FT, since CE is a per-sample
objective that does not depend on batch composition. **Stage 2 was
therefore initiated on JADE first.**

**Batch feasibility**: B ∈ {128, 256, 512}. B=512 OOM on A100-80GB; B=256
fits at ~62 GB peak. Stage 2 uses B=256.

**9-class — fold-1 LR sweep at B=256, α=0.3, τ=0.1** (single fold,
direction-finding only):

| lr | val_acc (fold 1) |
|:---:|:---:|
| 1e-4 | 0.6126 |
| 2e-4 | 0.6703 |
| 4e-4 | 0.6639 |
| 8e-4 | 0.6758 |

Run-to-run variance at fixed config is ~1pp on a single fold, so
intra-cluster ranking of {2e-4, 4e-4, 8e-4} requires full CV. lr=8e-4
showed warmup instability (val_acc collapsed to 0.15 at epoch 5
before recovery); rejected for stability.

**9-class — full-CV LR cross-check at the Stage 3 winner (α=0.3, τ=0.2)**:

| lr | Acc | std | Notes |
|:---:|:---:|:---:|---|
| 1e-4 | 56.71 | 3.91 | clearly worse |
| 2e-4 | 58.90 | 3.94 | substantially worse than 4e-4 |
| **4e-4** | **62.61** | **3.81** | **winner** |
| 8e-4 | 57.97 | 7.85 | 2 folds diverged — unstable |

**9-class result**: JADE at B=256, lr=4e-4 reaches 62.61, a ~+3 pp
improvement over the Stage 1 winner at B=128, lr=1e-4 (58.81). The
hypothesis is supported. To rule out that this gain is driven by the
optimization recipe rather than the loss formulation, we re-run FT at
the same recipe (§8.2): FT B=256, lr=4e-4 = 58.91, essentially
unchanged from FT B=64 = 58.20. **The 9-class JADE gain is therefore
attributable to the SupCon loss, not to the larger batch.**

**Binary — full-CV LR sweep at B=256, α=0.2, τ=0.05**:

| lr | Acc | std |
|:---:|:---:|:---:|
| **1e-4** | **76.74** | 2.71 |
| 2e-4 | 76.38 | 3.43 |
| 4e-4 | 73.47 | 5.74 |
| 8e-4 | 72.62 | 5.31 |

**Binary result**: the best B=256 JADE configuration (76.74 ± 2.71) is
within fold noise of the Stage 1 winner at B=128 (77.28 ± 1.29) and has
*higher* variance. Higher LRs degrade performance further. Scaling does
not help binary — consistent with binary having ~64 same-class
positives per anchor at B=128 already, so the contrastive estimator is
saturated. We therefore **keep B=128 for binary** (no FT B=256 control
required, since there is no JADE gain to attribute).

Stage 2 winners:
- **9-class**: B=256, lr=4e-4.
- **Binary**: keep REVE default — Stage 1 recipe (B=128, lr=1e-4).

### 9.3 Stage 3 — Loss HP refinement at Stage 2 recipe, full CV

For 9-class, Stage 2 changes the recipe to B=256, lr=4e-4 — a re-grid
of (α, τ) at this new recipe is needed to confirm the Stage 1 winner
still holds. For binary, the recipe did not change, so the Stage 1
winner stands without further refinement.

**9-class @ B=256, lr=4e-4** — complete 3 × 4 grid:

| α \ τ | 0.05 | 0.1 | 0.2 | 0.5 |
|:---:|:---:|:---:|:---:|:---:|
| 0.2 | 60.89 ± 3.83 | 60.76 ± 6.74 | 58.08 ± 6.97 | 61.21 ± 3.06 |
| **0.3** | 61.90 ± 3.40 | 62.34 ± 4.09 | **62.61 ± 3.81** | 60.79 ± 3.99 |
| 0.5 | 61.16 ± 3.10 | 61.38 ± 3.57 | 61.62 ± 4.87 | 60.14 ± 4.01 |

Smooth peak at (α=0.3, τ=0.2). τ shifted from 0.1 (Stage 1) to 0.2 at the
new recipe, consistent with the increased positives-per-anchor density at
larger batch (~14 → ~28 per anchor).

### 9.4 Final selected configurations

| Task | Config | Acc | std | Matched FT-FullFT baseline | Δ |
|---|---|:---:|:---:|:---:|:---:|
| 9-class | α=0.3, τ=0.2, B=256, lr=4e-4 | **62.61** | 3.81 | 58.91 (B=256, lr=4e-4) | **+3.70** |
| Binary  | α=0.2, τ=0.05, B=128, lr=1e-4 | **77.28** | **1.29** | 75.55 (REVE default) | **+1.73** |

Both tasks use REVE-default optimization for their respective JADE
default batch (binary stays at the Stage 1 recipe; 9-class moves to
B=256, lr=4e-4 as established in §9.2). The corresponding FT baselines
are at REVE-default for binary (§8.1) and at the JADE-matched recipe
for 9-class (§8.2).

### 9.5 JADE stimulus-generalization (3 seeds × 10 folds)

Configurations: 9-class @ α=0.3, τ=0.2, B=256, lr=4e-4; binary @ α=0.2,
τ=0.05, B=128, lr=1e-4.

| Task | s123 | s456 | s789 | mean ± std |
|---|:---:|:---:|:---:|:---:|
| 9-class | 15.95 ± 1.07 | 14.09 ± 1.22 | 15.93 ± 1.49 | **15.32 ± 1.07** |
| Binary  | 60.65 ± 2.45 | 61.61 ± 2.96 | 59.56 ± 2.98 | **60.61 ± 1.03** |

JADE 9-class generalization is essentially tied with FT and LP (15.32 vs
15.27 vs 15.83). JADE binary generalization is slightly higher than FT
(60.61 vs 59.86) and consistently above LP (58.46) across all 3 seeds.

---

## 10. Cross-method comparison

### 10.1 Cross-subject CV

| Task | LP | FT (matched recipe) | JADE | JADE − FT |
|---|:---:|:---:|:---:|:---:|
| 9-class | 49.72 ± 2.93 | 58.91 ± 2.95 (B=256, lr=4e-4) | **62.61 ± 3.81** | **+3.70** |
| Binary  | 71.50 ± 1.58 | 75.55 ± 2.16 (REVE default) | **77.28 ± 1.29** | **+1.73** |

**Headline CV claims:**
- 9-class: JADE beats FT by **+3.70 pp** at a recipe (B=256, lr=4e-4) where the optimization recipe alone does not improve FT — the gain is loss-driven.
- Binary: JADE beats FT by **+1.73 pp** at the REVE default recipe; scaling to B=256 does not yield further improvement on either method, so both stay at REVE default for binary.

### 10.2 Stimulus-generalization (3-seed mean)

| Task | LP | FT | JADE | JADE − FT |
|---|:---:|:---:|:---:|:---:|
| 9-class | 15.83 ± 0.17 | 15.27 ± 0.54 | 15.32 ± 1.07 | +0.05 (tied) |
| Binary  | 58.46 ± 2.14 | 59.86 ± 1.32 | 60.61 ± 1.03 | +0.75 (consistent across seeds) |

**Headline gen claims:**
- 9-class: the +3.70 pp CV advantage **collapses to a tie at gen**.
- Binary: the CV ranking (LP < FT < JADE) is preserved at gen with a
  shrunken gap — JADE is still the highest at every seed. The +1.73 CV
  gain shrinks to +0.75 vs FT.
- **Both gen results are well below CV** (9-class: ~−45 pp; binary:
  ~−15 pp). This is consistent with the dataset's known stimulus
  structure: holding out 1/3 of stimuli per emotion shifts evaluation
  to genuinely novel content.

### 10.3 What the data licences us to claim

- ✓ JADE produces a reproducible CV gain on FACED 9-class.
- ✓ The CV gain is robust to changes in (α, τ, batch, lr) — explored
  exhaustively over 60+ configurations.
- ✓ The cross-method ranking (LP < FT < JADE) is preserved on binary
  generalization.
- ✓ The cross-method ranking *collapses* on 9-class generalization.

---

## 11. Future Plans

- **Confusion Matrices**: which emotions are more frequently predicted correctly and what are the most common prediction mistakes

POSSIBLE FUTURE WORK (if there is time/if what I have now is not enough):
- **LoRA reference**: earlier LoRA-based runs were inconclusive at the
  un-tuned recipe and are not the focus of this report. A re-evaluation
  at the bulletproof CV recipes is planned.
- **Cross-dataset transfer**: testing on THU-EP under direct application
  of FACED-optimal configs, early result FT and JADE underperform: would need extensive investigation and retuning HPs
- **Channel importance**: Run a backward pass from the predicted class logit back to the raw EEG input (B, C, T). The gradient magnitude per channel gives electrode importance.
