# Results Brief — for the Report's "Results" Section

This document is a **packaging of every result artifact currently available** under
`main-results/` and `src/visualization/`. All numbers below are
extracted directly from the JSON summaries committed to `main-results/`.

---

## 1. Experimental setup (one-paragraph recap, for context)

- **Dataset.** FACED, 123 subjects, 32 EEG channels, 28 emotional video stimuli,
  200 Hz. Per-subject data shape after preprocessing: `(28, 32, 6000)`. Windows
  of 10 seconds (`window_size=2000`, `stride=2000`, non-overlapping).
- **Tasks.** Two classification tasks share the same windows:
  - **9-class** — 9 emotional categories (Anger, Disgust, Fear, Sadness,
    Neutral, Amusement, Inspiration, Joy, Tenderness).
  - **Binary** — valence: Negative (anger/disgust/fear/sadness) vs Positive
    (amusement/inspiration/joy/tenderness). Neutral stimuli are dropped.
- **Evaluation.** 10-fold cross-subject CV (KFold, shuffle=True, seed=42). Each
  fold trains on ~110 subjects and validates on ~13 held-out subjects. Every
  subject appears in exactly one validation fold, giving one accuracy per
  subject for N = 123 subjects.
- **Three methods compared.**
  - **LP** (Linear Probing) — REVE encoder frozen, only `cls_query_token` and a
    linear head are trained.
  - **SFT** (Supervised Fine-Tuning) — full fine-tuning of the entire encoder
    + head with cross-entropy loss only.
  - **JADE** — our method. Same backbone as SFT, but trained with a joint
    objective `L = α·CE + (1−α)·SupCon`, where SupCon is the supervised
    contrastive loss of Khosla et al. (2020). A projection head (used only at
    training time for the contrastive loss) sits on top of the encoder.
- **Hyperparameters of the reported JADE configurations** (selected from full
  sweep, see `docs/jade_hp_sweep.md`):
  - 9-class: α=0.3, τ=0.2, batch=256, lr=4e-4, SupCon repr = "context".
  - Binary:  α=0.2, τ=0.05, batch=128, lr=1e-4, SupCon repr = "context".
  - The matching SFT baseline was re-run at the *same* batch/LR recipe so the
    JADE-vs-SFT comparison is at parity (no LR confound).
- **Inference / metric definitions.** All headline numbers come from a
  post-hoc subject-wise inference pass (`src/inference/inference_subject_wise.py`).
  The script reloads each fold's best checkpoint, runs the held-out subjects
  through the frozen model with `shuffle=False`, and computes per-window
  predictions which are then aggregated:
  - **Subject-wise accuracy** = per-subject mean accuracy over its windows.
    The reported headline mean/std is over the 123 subject accuracies (this is
    the standard cross-subject literature metric).
  - **Macro F1, macro AUROC** = unweighted average across classes; AUROC is
    one-vs-rest.
  - **Per-class P / R / F1** = the standard sklearn computation per class on
    the pooled val windows.

---

## 2. Headline results table

This is the single most important table for the Results section. Same metrics
for all three methods on both tasks. All accuracies are subject-wise mean ±
std (N = 123 subjects).

| Method | Task    | Mean acc           | Macro F1   | Macro AUROC | Min subj   | Max subj   |
|--------|---------|--------------------|------------|-------------|------------|------------|
| LP     | 9-class | 50.27 ± 13.69 %    | 50.45 %    | 83.35 %     | 10.71 %    | 85.71 %    |
| SFT    | 9-class | 58.52 ± 14.01 %    | 58.93 %    | 88.52 %     |  7.14 %    | 85.71 %    |
| JADE   | 9-class | **62.03 ± 14.70 %**| **62.51 %**| **90.18 %** | 10.71 %    | **86.90 %**|
| LP     | binary  | 71.64 ±  9.29 %    | 71.61 %    | 77.07 %     | 50.00 %    | 94.44 %    |
| SFT    | binary  | 75.52 ±  7.76 %    | 75.51 %    | 81.82 %     | 50.00 %    | 93.06 %    |
| JADE   | binary  | **76.32 ±  8.12 %**| **76.32 %**| **82.65 %** | **52.78 %**| **97.22 %**|

**Headline improvements (all numbers in percentage points):**

| Task     | SFT − LP  | JADE − SFT   | JADE − LP  |
|----------|-----------|--------------|------------|
| 9-class  | +8.25 pp  | **+3.51 pp** | +11.76 pp  |
| Binary   | +3.88 pp  | **+0.80 pp** | +4.69 pp   |

**Reading the table.** Fine-tuning gives the largest absolute gain over the
linear probe in both tasks (+8.25 pp on 9-class, +3.88 pp on binary). On top of
that, adding the SupCon objective (JADE) recovers an additional +3.51 pp on
9-class but only +0.80 pp on binary — a finding that is central to the
discussion below.

---

## 3. Per-class metrics (precision / recall / F1)

### 3.1 JADE 9-class — full table

From `src/visualization/jade_9-class/figures/per_class_metrics.csv`:

| Class        | Precision  | Recall     | F1         |
|--------------|------------|------------|------------|
| Anger        | 63.50 %    | 63.96 %    | 63.73 %    |
| Disgust      | 61.67 %    | 61.34 %    | 61.50 %    |
| Fear         | 66.41 %    | 63.05 %    | 64.69 %    |
| Sadness      | 48.30 %    | 48.78 %    | 48.54 %    |
| Neutral      | 49.97 %    | 54.74 %    | 52.25 %    |
| Amusement    | 70.18 %    | 70.37 %    | 70.28 %    |
| Inspiration  | 63.76 %    | 64.68 %    | 64.22 %    |
| Joy          | 69.27 %    | 64.95 %    | 67.04 %    |
| Tenderness   | 71.95 %    | 68.83 %    | 70.36 %    |
| **Macro avg**| **62.78 %**| **62.30 %**| **62.51 %**|

Tenderness (70.36 %), Amusement (70.28 %) and Joy (67.04 %) are the easiest
classes; Sadness (48.54 %) and Neutral (52.25 %) are the hardest.

### 3.2 JADE binary — full table

From `src/visualization/jade_binary/figures/per_class_metrics.csv`:

| Class        | Precision  | Recall     | F1         |
|--------------|------------|------------|------------|
| Negative     | 76.46 %    | 76.06 %    | 76.26 %    |
| Positive     | 76.19 %    | 76.58 %    | 76.38 %    |
| **Macro avg**| **76.32 %**| **76.32 %**| **76.32 %**|

Performance is nearly symmetric across the two classes — no positive/negative
bias.

### 3.3 Per-class F1 — comparison of all three methods

This is what the figure `per_class_f1_bars_with_lp.pdf` displays graphically.

**9-class:**

| Class       | LP        | SFT       | JADE      | Δ(JADE−SFT)  |
|-------------|-----------|-----------|-----------|--------------|
| Anger       | 50.36 %   | 60.65 %   | 63.73 %   | **+3.08 pp** |
| Disgust     | 51.87 %   | 56.13 %   | 61.50 %   | **+5.37 pp** |
| Fear        | 51.77 %   | 60.83 %   | 64.69 %   | **+3.86 pp** |
| Sadness     | 38.54 %   | 47.64 %   | 48.54 %   |  +0.90 pp    |
| Neutral     | 45.69 %   | 49.70 %   | 52.25 %   |  +2.55 pp    |
| Amusement   | 56.54 %   | 66.79 %   | 70.28 %   |  +3.49 pp    |
| Inspiration | 46.88 %   | 58.83 %   | 64.22 %   | **+5.39 pp** |
| Joy         | 56.72 %   | 63.85 %   | 67.04 %   |  +3.19 pp    |
| Tenderness  | 55.64 %   | 65.95 %   | 70.36 %   |  +4.41 pp    |
| **Macro avg** | **50.45 %** | **58.93 %** | **62.51 %** | **+3.58 pp** |

Every class improves under JADE relative to SFT on 9-class. The largest gains
are on **Inspiration (+5.39 pp)**, **Disgust (+5.37 pp)** and
**Tenderness (+4.41 pp)**. Sadness gains the least (+0.90 pp) — it is also the
hardest class in absolute terms.

**Binary:**

| Class    | LP        | SFT       | JADE      | Δ(JADE−SFT)  |
|----------|-----------|-----------|-----------|--------------|
| Negative | 70.68 %   | 75.90 %   | 76.26 %   | +0.36 pp     |
| Positive | 72.53 %   | 75.13 %   | 76.38 %   | +1.25 pp     |
| **Macro avg** | **71.61 %** | **75.51 %** | **76.32 %** | **+0.81 pp** |

Per-class gains on binary are small and roughly symmetric, consistent with the
near-flat headline accuracy gain.

---

## 4. Confusion matrices (JADE)

Row-normalized percentages (each row sums to 100 %, diagonal = recall).

### 4.1 9-class

|              | Anger     | Disgust   | Fear      | Sadness   | Neutral   | Amuse     | Inspir    | Joy       | Tender    |
|--------------|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|
| **Anger**    | **63.96** | 4.79      | 3.43      | 5.51      | 8.58      | 2.80      | 4.70      | 3.25      | 2.98      |
| **Disgust**  | 6.14      | **61.34** | 2.89      | 7.14      | 11.29     | 2.44      | 3.61      | 2.98      | 2.17      |
| **Fear**     | 4.16      | 2.62      | **63.05** | 6.14      | 8.49      | 5.96      | 3.07      | 2.62      | 3.88      |
| **Sadness**  | 4.97      | 7.23      | 6.05      | **48.78** | 16.17     | 2.80      | 5.87      | 4.61      | 3.52      |
| **Neutral**  | 6.30      | 7.18      | 4.34      | 10.23     | **54.74** | 4.00      | 5.28      | 4.34      | 3.59      |
| **Amusement**| 2.17      | 2.62      | 3.34      | 5.06      | 6.87      | **70.37** | 3.43      | 3.16      | 2.98      |
| **Inspiration** | 4.34   | 4.07      | 3.07      | 5.24      | 7.86      | 3.25      | **64.68** | 4.43      | 3.07      |
| **Joy**      | 4.25      | 4.97      | 3.07      | 4.79      | 7.05      | 3.34      | 4.16      | **64.95** | 3.43      |
| **Tenderness**| 2.35     | 2.26      | 4.25      | 4.70      | 6.78      | 3.97      | 4.88      | 1.99      | **68.83** |

**Observations to highlight in the prose:**
1. **All diagonal entries are above chance (11.11 %) by a wide margin** — the
   weakest class still hits 48.78 %.
2. **Sadness is the hardest** (48.78 % recall) and gets confused most often with
   Neutral (16.17 %).
3. **Neutral attracts errors from most other classes** — it is the off-diagonal
   "magnet" column (the 8–16 % range in column Neutral across rows). This is
   consistent with Neutral being an ambiguous middle category.
4. **Within-valence confusions are mild but present** — e.g. Disgust↔Anger
   (~6 %), Amusement↔Joy (~3 %). The model does not collapse positive emotions
   onto each other.

### 4.2 Binary

|              | Negative  | Positive  |
|--------------|----------:|----------:|
| **Negative** | **76.06** | 23.94     |
| **Positive** | 23.42     | **76.58** |

Errors are symmetric (23.94 ↔ 23.42): no valence bias.

---

## 5. Per-subject results (cross-subject generalization)

### 5.1 Distribution statistics (JADE)

| Task    | N   | Mean       | Std    | Median     | Min        | Max        | Q25        | Q75        | Below chance |
|---------|-----|------------|--------|------------|------------|------------|------------|------------|--------------|
| 9-class | 123 | 62.03 %    | 14.70  | 64.29 %    | 10.71 %    | 86.90 %    | 52.38 %    | 72.62 %    | 1 / 123      |
| Binary  | 123 | 76.32 %    |  8.12  | 77.78 %    | 52.78 %    | 97.22 %    | 70.83 %    | 81.94 %    | 0 / 123      |

**Key observations:**
- The interquartile range is wide (~20 pp for 9-class, ~11 pp for binary):
  individual differences dominate. This is the *cross-subject EEG problem*
  that the chosen evaluation protocol is designed to expose.
- Only **1 subject out of 123 is below chance on 9-class**, and **none below
  chance on binary** — the model generalizes meaningfully to essentially every
  held-out subject.
- The headline mean is **slightly lower than the median** on both tasks
  (62.03 % vs 64.29 % on 9-class; 76.32 % vs 77.78 % on binary), so
  distributions are slightly left-skewed by a tail of harder subjects.

### 5.2 Paired comparison — wins, losses, mean delta

| Comparison              | N   | JADE wins | JADE loses | Ties | Mean Δ         |
|-------------------------|-----|-----------|------------|------|----------------|
| 9-class, JADE vs SFT    | 123 | **88**    | 22         | 13   | **+3.51 pp**   |
| 9-class, JADE vs LP     | 123 | **113**   | 6          | 4    | **+11.76 pp**  |
| Binary,  JADE vs SFT    | 123 | 61        | 49         | 13   | +0.80 pp       |
| Binary,  JADE vs LP     | 123 | **91**    | 26         | 6    | **+4.69 pp**   |

**Reading these counts:**
- On **9-class**, JADE outperforms SFT on **88 of 123 subjects (71.54 %)** —
  the improvement is broad-based, not driven by a handful of outliers.
- On **binary**, the win/loss split is much closer to even (61/49), consistent
  with the marginal +0.80 pp mean gain.
- Against LP, JADE wins on **91.87 % of subjects in 9-class** (113/123) and
  **73.98 % of subjects in binary** (91/123).

---

## 6. Figures available (with what each one shows)

All paths are relative to repo root. Figures are vector PDFs ready for the
report. Each figure has a `--title` flag and was generated *without* titles
(publication style); titles can be added by re-running the corresponding
script with `--title`.

### Per-task figures (8 total, 4 per task)

| File | Task | What it shows | Use in the report |
|------|------|---------------|-------------------|
| `src/visualization/jade_9-class/figures/confusion_matrix.pdf` | 9-class | JADE confusion matrix, row-normalized percentages. Power-norm colormap (γ=0.55) for off-diagonal contrast. | Main per-class error analysis (cite alongside §4.1). |
| `src/visualization/jade_binary/figures/confusion_matrix.pdf`  | binary  | JADE binary CM, row-normalized. | Shows balanced errors (cite alongside §4.2). |
| `src/visualization/jade_9-class/figures/subject_histogram.pdf` | 9-class | 123 subjects sorted ascending by accuracy. Green dashed line = mean (62.03 %), red dotted line = chance (11.11 %). Side bar shows mean ± std. | The cross-subject variability figure. Use alongside §5.1. |
| `src/visualization/jade_binary/figures/subject_histogram.pdf`  | binary | Same layout for binary (chance = 50.00 %). | Same role for binary. |
| `src/visualization/jade_9-class/figures/per_class_f1_bars.pdf` | 9-class | Per-class F1 bars JADE vs SFT side-by-side + Macro avg group. Δ(JADE−SFT) annotated in pp above each pair. | Visual summary of §3.3 (9-class). |
| `src/visualization/jade_binary/figures/per_class_f1_bars.pdf`  | binary  | Same for binary. | §3.3 (binary). |
| `src/visualization/jade_9-class/figures/per_class_f1_bars_with_lp.pdf` | 9-class | Same as above but with 3 bars per class (LP / SFT / JADE). Δ annotation still shows JADE − SFT. | Optional alternative — broader context showing how much of the gain is FT vs how much is SupCon on top. |
| `src/visualization/jade_binary/figures/per_class_f1_bars_with_lp.pdf`  | binary  | Same for binary. | Same. |
| `src/visualization/jade_9-class/figures/paired_subject_scatter.pdf` | 9-class | Scatter: SFT subject accuracy (x) vs JADE subject accuracy (y), one dot per subject. y=x reference line, inline stats box (wins/losses/ties/meanΔ). | Visual proof that the JADE-vs-SFT gain is broad-based, not outlier-driven. Use alongside §5.2. |
| `src/visualization/jade_binary/figures/paired_subject_scatter.pdf`  | binary | Same for binary. | Same. |
| `src/visualization/jade_9-class/figures/paired_subject_scatter_with_lp.pdf` | 9-class | Two panels side-by-side: JADE vs SFT (left) and JADE vs LP (right). | Optional — confirms that LP is a much weaker baseline. |
| `src/visualization/jade_binary/figures/paired_subject_scatter_with_lp.pdf`  | binary | Same for binary. | Same. |

### Tables already typeset

- `src/visualization/jade_9-class/figures/per_class_metrics.tex` — LaTeX
  booktabs table of the 9-class per-class metrics shown in §3.1.
- `src/visualization/jade_binary/figures/per_class_metrics.tex` — same for binary.
- Companion `.csv` files in the same directories for direct re-import.

---

## 7. Narrative for the Results section

The numbers above support **one main story arc with two task-specific
sub-narratives**. The writing agent should structure the prose around these:

### 7.1 Headline finding (one short paragraph)

JADE outperforms both baselines on both tasks. On the harder 9-class problem
it lifts subject-wise accuracy from 58.52 % (SFT) and 50.27 % (LP) to
**62.03 %** — gains of +3.51 pp and +11.76 pp respectively. On the binary
task the picture is different: JADE reaches **76.32 %**, only +0.80 pp over
SFT but +4.69 pp over LP. **The contribution of supervised contrastive
learning depends on the task.**

### 7.2 Sub-narrative A — 9-class: SupCon helps broadly and per class

- Improvement is **broad-based, not outlier-driven**: JADE beats SFT on 88
  out of 123 subjects (71.54 %; cf. §5.2).
- Improvement is **broad across classes**: all 9 classes improve under JADE
  vs SFT; the largest gains are on Inspiration (+5.39 pp), Disgust (+5.37 pp)
  and Tenderness (+4.41 pp) (cf. §3.3).
- Confusion patterns suggest that the **hardest class is Sadness** (48.78 %
  recall) and the **principal error mode is confusion with Neutral**
  (16.17 % off-diagonal); Neutral itself acts as an error magnet across rows
  (cf. §4.1).
- Cross-subject variability is high (std = 14.70 pp, range 10.71 % – 86.90 %),
  but only 1/123 subjects fall below chance — the model generalizes broadly
  (cf. §5.1).

### 7.3 Sub-narrative B — binary: SupCon's value is small once SFT is tuned

- On binary, JADE improves over SFT by only +0.80 pp (76.32 % vs 75.52 %).
- The win/loss split is essentially even (61 wins, 49 losses, 13 ties; §5.2):
  on roughly 40 % of subjects JADE underperforms SFT.
- Per-class F1 gains are small (+0.36 / +1.25 pp) and symmetric across
  Negative / Positive (cf. §3.3, §4.2).
- **Interpretation.** The binary task is comparatively saturated. SupCon's
  mechanism (pulling same-class samples together, pushing different-class
  apart) provides the most leverage when there are *many* classes with
  competing intra-class structure to disambiguate (9-class). With only two
  classes, CE alone already finds a near-optimal embedding for the same
  encoder capacity, so the contrastive term has little room to add value.
  This task-dependence of contrastive auxiliary objectives is the central
  methodological finding of the thesis.

### 7.4 Optional points worth mentioning if space allows

- **Fairness of comparison.** The SFT baseline against which JADE is measured
  was re-run at the *same* batch/LR recipe as the JADE configuration; the
  reported gain does not benefit from an under-tuned baseline. (This is a
  methodological norm of the thesis — see `CLAUDE.md` "thesis status".)
- **AUROC.** Reported in the headline table for completeness (macro AUROC
  rises from 83.35 % (LP) to 88.52 % (SFT) to 90.18 % (JADE) on 9-class). No
  ROC figure was produced — for a 9-class problem the curves overlap heavily
  and add little beyond the single AUROC number.
- **The "averaged accuracy" side bar** in the subject-histogram figures is
  the mean ± std of subject accuracies for that task (matching the
  headline-table row for JADE on that task), provided so the figure can be
  read standalone.

---

## 8. Verbatim numbers cheat-sheet (copy-paste-safe)

For the report writer to quote without recomputation:

- JADE 9-class: **62.03 ± 14.70 %** subject-wise; macro F1 **62.51 %**; macro AUROC **90.18 %**.
- JADE binary:  **76.32 ±  8.12 %** subject-wise; macro F1 **76.32 %**; macro AUROC **82.65 %**.
- SFT 9-class:  **58.52 ± 14.01 %**; macro F1 **58.93 %**; macro AUROC **88.52 %**.
- SFT binary:   **75.52 ±  7.76 %**; macro F1 **75.51 %**; macro AUROC **81.82 %**.
- LP 9-class:   **50.27 ± 13.69 %**; macro F1 **50.45 %**; macro AUROC **83.35 %**.
- LP binary:    **71.64 ±  9.29 %**; macro F1 **71.61 %**; macro AUROC **77.07 %**.
- 9-class gain (JADE − SFT): **+3.51 pp** accuracy / **+3.58 pp** macro F1.
- Binary gain  (JADE − SFT): **+0.80 pp** accuracy / **+0.81 pp** macro F1.
- 9-class subject wins JADE > SFT: **88 / 123** (71.54 %).
- Binary  subject wins JADE > SFT: **61 / 123** (49.59 %).
- Subjects below chance: **1 / 123 on 9-class**, **0 / 123 on binary** (JADE).
