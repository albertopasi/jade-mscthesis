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
  sweep, see `docs/jade_hp_methodology.md`):
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
  - **Window-wise accuracy** = pooled over all val windows, ignoring subject
    membership. Reported for completeness.
  - **Macro F1, macro AUROC** = unweighted average across classes; AUROC is
    one-vs-rest.
  - **Per-class P / R / F1** = the standard sklearn computation per class on
    the pooled val windows.

---

## 2. Headline results table

This is the single most important table for the Results section. Same metrics
for all three methods on both tasks. All accuracies are subject-wise mean ±
std (N = 123 subjects).

| Method | Task    | Mean acc        | Window acc | Macro F1 | Macro AUROC | Min subj | Max subj |
|--------|---------|-----------------|------------|----------|-------------|----------|----------|
| LP     | 9-class | 50.27 ± 13.69 % | 50.27 %    | 0.5045   | 0.8335      | 10.71 %  | 85.71 %  |
| SFT    | 9-class | 58.52 ± 14.01 % | 58.52 %    | 0.5893   | 0.8852      |  7.14 %  | 85.71 %  |
| JADE   | 9-class | **62.03 ± 14.70 %** | **62.03 %** | **0.6251** | **0.9018** | 10.71 %  | **86.90 %** |
| LP     | binary  | 71.64 ±  9.29 % | 71.64 %    | 0.7161   | 0.7707      | 50.00 %  | 94.44 %  |
| SFT    | binary  | 75.52 ±  7.76 % | 75.52 %    | 0.7551   | 0.8182      | 50.00 %  | 93.06 %  |
| JADE   | binary  | **76.32 ±  8.12 %** | **76.32 %** | **0.7632** | **0.8265** | **52.78 %** | **97.22 %** |

**Headline improvements (all numbers in percentage points):**

| Task     | SFT − LP  | JADE − SFT | JADE − LP |
|----------|-----------|------------|-----------|
| 9-class  | +8.25 pp  | **+3.51 pp** | +11.76 pp |
| Binary   | +3.88 pp  | **+0.80 pp** | +4.69 pp  |

**Reading the table.** Fine-tuning gives the largest absolute gain over the
linear probe in both tasks (+8.25 pp on 9-class, +3.88 pp on binary). On top of
that, adding the SupCon objective (JADE) recovers an additional +3.51 pp on
9-class but only +0.80 pp on binary — a finding that is central to the
discussion below.

---

## 3. Per-class metrics (precision / recall / F1)

### 3.1 JADE 9-class — full table

From `src/visualization/jade_9-class/figures/per_class_metrics.csv`:

| Class        | Precision | Recall | F1    | Support |
|--------------|-----------|--------|-------|---------|
| Anger        | 0.635     | 0.640  | 0.637 | 1107    |
| Disgust      | 0.617     | 0.613  | 0.615 | 1107    |
| Fear         | 0.664     | 0.630  | 0.647 | 1107    |
| Sadness      | 0.483     | 0.488  | 0.485 | 1107    |
| Neutral      | 0.500     | 0.547  | 0.522 | 1476    |
| Amusement    | 0.702     | 0.704  | 0.703 | 1107    |
| Inspiration  | 0.638     | 0.647  | 0.642 | 1107    |
| Joy          | 0.693     | 0.649  | 0.670 | 1107    |
| Tenderness   | 0.720     | 0.688  | 0.704 | 1107    |
| **Macro avg**| **0.628** | **0.623** | **0.625** | 10332 |

Tenderness (0.704), Amusement (0.703) and Joy (0.670) are the easiest
classes; Sadness (0.485) and Neutral (0.522) are the hardest. The Neutral row
has higher support (1476 vs 1107) because that class is *not* dropped in
9-class mode (it is only dropped in binary mode).

### 3.2 JADE binary — full table

From `src/visualization/jade_binary/figures/per_class_metrics.csv`:

| Class        | Precision | Recall | F1    | Support |
|--------------|-----------|--------|-------|---------|
| Negative     | 0.765     | 0.761  | 0.763 | 4428    |
| Positive     | 0.762     | 0.766  | 0.764 | 4428    |
| **Macro avg**| **0.763** | **0.763** | **0.763** | 8856 |

Performance is nearly symmetric across the two classes — no positive/negative
bias.

### 3.3 Per-class F1 — comparison of all three methods

This is what the figure `per_class_f1_bars_with_lp.pdf` displays graphically.

**9-class:**

| Class       | LP    | SFT   | JADE  | Δ(JADE−SFT) |
|-------------|-------|-------|-------|-------------|
| Anger       | 0.504 | 0.607 | 0.637 | **+3.08** pp |
| Disgust     | 0.519 | 0.561 | 0.615 | **+5.37** pp |
| Fear        | 0.518 | 0.608 | 0.647 | **+3.86** pp |
| Sadness     | 0.385 | 0.476 | 0.485 |  +0.90 pp |
| Neutral     | 0.457 | 0.497 | 0.522 |  +2.55 pp |
| Amusement   | 0.565 | 0.668 | 0.703 |  +3.49 pp |
| Inspiration | 0.469 | 0.588 | 0.642 | **+5.39** pp |
| Joy         | 0.567 | 0.638 | 0.670 |  +3.19 pp |
| Tenderness  | 0.556 | 0.659 | 0.704 |  +4.41 pp |
| **Macro avg** | **0.504** | **0.589** | **0.625** | **+3.58 pp** |

Every class improves under JADE relative to SFT on 9-class. The largest gains
are on **Inspiration (+5.39)**, **Disgust (+5.37)** and **Tenderness (+4.41)**.
Sadness gains the least (+0.90) — it is also the hardest class in absolute
terms.

**Binary:**

| Class    | LP    | SFT   | JADE  | Δ(JADE−SFT) |
|----------|-------|-------|-------|-------------|
| Negative | 0.707 | 0.759 | 0.763 | +0.36 pp |
| Positive | 0.725 | 0.751 | 0.764 | +1.25 pp |
| **Macro avg** | **0.716** | **0.755** | **0.763** | **+0.81 pp** |

Per-class gains on binary are small and roughly symmetric, consistent with the
near-flat headline accuracy gain.

---

## 4. Confusion matrices (JADE)

Row-normalized percentages (each row sums to 100 %, diagonal = recall).

### 4.1 9-class

|              | Anger | Disgust | Fear | Sadness | Neutral | Amuse | Inspir | Joy  | Tender |
|--------------|------:|--------:|-----:|--------:|--------:|------:|-------:|-----:|-------:|
| **Anger**    | **64.0** | 4.8 | 3.4 | 5.5 | 8.6 | 2.8 | 4.7 | 3.3 | 3.0 |
| **Disgust**  | 6.1 | **61.3** | 2.9 | 7.1 | 11.3 | 2.4 | 3.6 | 3.0 | 2.2 |
| **Fear**     | 4.2 | 2.6 | **63.1** | 6.1 | 8.5 | 6.0 | 3.1 | 2.6 | 3.9 |
| **Sadness**  | 5.0 | 7.2 | 6.1 | **48.8** | 16.2 | 2.8 | 5.9 | 4.6 | 3.5 |
| **Neutral**  | 6.3 | 7.2 | 4.3 | 10.2 | **54.7** | 4.0 | 5.3 | 4.3 | 3.6 |
| **Amusement**| 2.2 | 2.6 | 3.3 | 5.1 | 6.9 | **70.4** | 3.4 | 3.2 | 3.0 |
| **Inspiration** | 4.3 | 4.1 | 3.1 | 5.2 | 7.9 | 3.3 | **64.7** | 4.4 | 3.1 |
| **Joy**      | 4.2 | 5.0 | 3.1 | 4.8 | 7.0 | 3.3 | 4.2 | **65.0** | 3.4 |
| **Tenderness**| 2.3 | 2.3 | 4.2 | 4.7 | 6.8 | 4.0 | 4.9 | 2.0 | **68.8** |

**Observations to highlight in the prose:**
1. **All diagonal entries are above chance (11.1 %) by a wide margin** — the
   weakest class still hits 48.8 %.
2. **Sadness is the hardest** (48.8 % recall) and gets confused most often with
   Neutral (16.2 %).
3. **Neutral attracts errors from most other classes** — it is the off-diagonal
   "magnet" column (the 8–16 % range in column Neutral across rows). This is
   consistent with Neutral being an ambiguous middle category.
4. **Within-valence confusions are mild but present** — e.g. Disgust↔Anger
   (~6 %), Amusement↔Joy (~3 %). The model does not collapse positive emotions
   onto each other.

### 4.2 Binary

|              | Negative | Positive |
|--------------|---------:|---------:|
| **Negative** | **76.1** | 23.9 |
| **Positive** | 23.4 | **76.6** |

Errors are symmetric (23.9 ↔ 23.4): no valence bias.

---

## 5. Per-subject results (cross-subject generalization)

### 5.1 Distribution statistics (JADE)

| Task    | N   | Mean    | Std   | Median  | Min     | Max     | Q25     | Q75     | Below chance |
|---------|-----|---------|-------|---------|---------|---------|---------|---------|--------------|
| 9-class | 123 | 62.03 % | 14.70 | 64.29 % | 10.71 % | 86.90 % | 52.38 % | 72.62 % | 1 / 123 |
| Binary  | 123 | 76.32 % |  8.12 | 77.78 % | 52.78 % | 97.22 % | 70.83 % | 81.94 % | 0 / 123 |

**Key observations:**
- The interquartile range is wide (~20 pp for 9-class, ~11 pp for binary):
  individual differences dominate. This is the *cross-subject EEG problem*
  that the chosen evaluation protocol is designed to expose.
- Only **1 subject out of 123 is below chance on 9-class**, and **none below
  chance on binary** — the model generalizes meaningfully to essentially every
  held-out subject.
- The headline mean is **higher than the median** on 9-class (62.03 vs 64.29)
  and roughly equal on binary (76.32 vs 77.78), so distributions are slightly
  left-skewed by a tail of harder subjects.

### 5.2 Paired comparison — wins, losses, mean delta

| Comparison              | N | JADE wins | JADE loses | Ties | Mean Δ      |
|-------------------------|---|-----------|------------|------|-------------|
| 9-class, JADE vs SFT    | 123 | **88** | 22 | 13 | **+3.51 pp** |
| 9-class, JADE vs LP     | 123 | **113** | 6 | 4 | **+11.76 pp** |
| Binary,  JADE vs SFT    | 123 | 61 | 49 | 13 | +0.80 pp |
| Binary,  JADE vs LP     | 123 | **91** | 26 | 6 | **+4.69 pp** |

**Reading these counts:**
- On **9-class**, JADE outperforms SFT on **88 of 123 subjects (72 %)** — the
  improvement is broad-based, not driven by a handful of outliers.
- On **binary**, the win/loss split is much closer to even (61/49), consistent
  with the marginal +0.80 pp mean gain.
- Against LP, JADE wins on **92 % of subjects in 9-class** and **74 % of
  subjects in binary**.

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
| `src/visualization/jade_9-class/figures/subject_histogram.pdf` | 9-class | 123 subjects sorted ascending by accuracy. Green dashed line = mean (62.0 %), red dotted line = chance (11.1 %). Side bar shows mean ± std. | The cross-subject variability figure. Use alongside §5.1. |
| `src/visualization/jade_binary/figures/subject_histogram.pdf`  | binary | Same layout for binary (chance = 50 %). | Same role for binary. |
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
  out of 123 subjects (72 %; cf. §5.2).
- Improvement is **broad across classes**: all 9 classes improve under JADE
  vs SFT; the largest gains are on Inspiration (+5.39 pp), Disgust (+5.37 pp)
  and Tenderness (+4.41 pp) (cf. §3.3).
- Confusion patterns suggest that the **hardest class is Sadness** (48.8 %
  recall) and the **principal error mode is confusion with Neutral** (16.2 %
  off-diagonal); Neutral itself acts as an error magnet across rows (cf. §4.1).
- Cross-subject variability is high (std = 14.7 pp, range 10.7 % – 86.9 %),
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
  rises from 0.834 (LP) to 0.885 (SFT) to 0.902 (JADE) on 9-class). No ROC
  figure was produced — for a 9-class problem the curves overlap heavily and
  add little beyond the single AUROC number.
- **The “averaged accuracy” side bar** in the subject-histogram figures is
  the mean ± std of subject accuracies for that task (matching the
  headline-table row for JADE on that task), provided so the figure can be
  read standalone.

---

## 8. Verbatim numbers cheat-sheet (copy-paste-safe)

For the report writer to quote without recomputation:

- JADE 9-class: **62.03 ± 14.70 %** subject-wise; macro F1 **0.6251**; macro AUROC **0.9018**.
- JADE binary:  **76.32 ±  8.12 %** subject-wise; macro F1 **0.7632**; macro AUROC **0.8265**.
- SFT 9-class:  **58.52 ± 14.01 %**; macro F1 **0.5893**; macro AUROC **0.8852**.
- SFT binary:   **75.52 ±  7.76 %**; macro F1 **0.7551**; macro AUROC **0.8182**.
- LP 9-class:   **50.27 ± 13.69 %**; macro F1 **0.5045**; macro AUROC **0.8335**.
- LP binary:    **71.64 ±  9.29 %**; macro F1 **0.7161**; macro AUROC **0.7707**.
- 9-class gain (JADE − SFT): **+3.51 pp** accuracy / **+3.58 pp** macro F1.
- Binary gain  (JADE − SFT): **+0.80 pp** accuracy / **+0.81 pp** macro F1.
- 9-class subject wins JADE > SFT: **88 / 123** (72 %).
- Binary  subject wins JADE > SFT: **61 / 123** (50 %).
- Subjects below chance: **1/123 on 9-class**, **0/123 on binary** (JADE).
