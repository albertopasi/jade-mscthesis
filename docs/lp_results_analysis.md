# Linear Probing Results Analysis

**Experiment:** REVE linear probing on FACED and THU-EP EEG emotion datasets

**Mode:** Official (frozen REVE encoder, trainable `cls_query_token` + linear head, `pooling=no`)

**Window:** 10 s (2000 pts at 200 Hz), non-overlapping

---

## 1. What Was Measured

Two evaluation strategies were run:

| Strategy | Description |
|---|---|
| **Cross-subject CV** | 10-fold KFold — model trains on ~90% of subjects, validates on ~10% of unseen subjects. Tests whether the model generalises across people. |
| **Stimulus generalisation** | Combines the subject split from standard CV with an additional stimulus split. Training uses `train_subjects × train_stimuli` (2/3 of stimuli per emotion); validation uses `val_subjects × val_stimuli` (held-out 1/3 of stimuli). Both the subjects and the stimuli are unseen at validation. Tests whether emotion representations transfer to new people exposed to new stimuli. |

Stimulus generalisation was run with 3 random seeds (123, 456, 789) to average out split randomness.

### Metrics explained

| Metric | What it measures | Notes |
|---|---|---|
| **Accuracy** | Fraction of windows correctly classified | Most intuitive; can be misleading if classes are imbalanced |
| **Balanced Accuracy** | Mean per-class recall | Corrects for class imbalance; matches accuracy when classes are balanced |
| **AUROC** | Area under the ROC curve (one-vs-rest, macro) | Measures ranking quality: how well the model separates classes in probability space, independent of threshold. 0.5 = random, 1.0 = perfect |
| **Weighted F1** | F1 averaged by class support | Combines precision and recall; more sensitive to class-specific errors than accuracy |

**Chance levels:**
- Binary (pos vs neg): **50%** accuracy / 0.50 AUROC
- 9-class: **11.1%** accuracy / 0.50 AUROC (9 balanced classes)

---

## 2. Cross-Subject Cross-Validation Results (10 folds)

| Dataset | Task | Accuracy | Bal. Accuracy | AUROC | Weighted F1 |
|---|---|---|---|---|---|
| FACED | Binary | **70.57% ± 1.56%** | 70.57% ± 1.56% | 76.83% ± 2.58% | 70.46% ± 1.65% |
| FACED | 9-class | **48.91% ± 2.86%** | 49.24% ± 2.56% | **84.16% ± 2.79%** | 48.37% ± 3.30% |
| THU-EP | Binary | **65.07% ± 3.31%** | 65.09% ± 3.31% | 71.32% ± 5.01% | 64.35% ± 3.82% |
| THU-EP | 9-class | **41.06% ± 2.06%** | 41.30% ± 1.90% | **79.99% ± 2.31%** | 40.46% ± 2.43% |

**Key observations:**

- Accuracy ≡ Balanced Accuracy in binary — the dataset is perfectly class-balanced (12 positive, 12 negative stimuli), so no skew.
- The **AUROC–accuracy gap** in 9-class is striking: 84% AUROC at only 49% accuracy (FACED), 80% AUROC at 41% accuracy (THU-EP). The model ranks the correct class highly even when it does not rank it first — the representations encode emotion structure well, but fine-grained boundaries are soft (see §4.3).
- THU-EP shows higher fold variance than FACED (binary std 3.31% vs 1.56%), likely due to fewer subjects (79 vs 123) giving each fold a smaller and less representative validation set.
- All folds ran the full 20 epochs — no early stopping triggered. Several folds had `best_epoch = 20`, indicating the training budget was too tight and more epochs would likely help.

---

## 3. Stimulus Generalisation Results (3 seeds: 123, 456, 789)

| Dataset | Task | Accuracy | Bal. Accuracy | F1 | vs CV Acc (Δ) |
|---|---|---|---|---|---|
| FACED | Binary | 58.46% ± 2.14% | 58.46% ± 2.14% | 56.86% ± 2.15% | −12.11 pp |
| FACED | 9-class | **15.83% ± 0.17%** | 15.83% ± 0.17% | 12.15% ± 0.83% | −33.08 pp |
| THU-EP | Binary | 55.59% ± 1.41% | 55.57% ± 1.35% | 53.21% ± 1.81% | −9.48 pp |
| THU-EP | 9-class | **17.50% ± 0.26%** | 17.48% ± 0.24% | 14.16% ± 0.63% | −23.56 pp |

**Key observations:**

- Binary tasks show meaningful seed variance (1.4–2.1%), meaning which stimuli are held out affects performance noticeably — some stimulus subsets provide a stronger training signal than others.
- 9-class tasks have near-zero seed variance (0.17–0.26%), meaning all seeds produce uniformly near-chance results. The collapse is systematic, not a matter of bad luck with a particular split.
- AUROC in 9-class generalisation drops to ~53–56% (barely above random), compared to ~80–84% in CV. The model's ranking ability also depends on stimulus-specific cues, not just emotion structure.

---

## 4. Cross-cutting Analysis

### 4.1 FACED vs THU-EP

| | Binary CV | 9-class CV |
|---|---|---|
| FACED | 70.57% | 48.91% |
| THU-EP | 65.07% | 41.06% |
| **Gap** | **5.50 pp** | **7.85 pp** |

FACED consistently outperforms THU-EP. Likely contributors:

1. **More subjects** — 123 vs 79 gives more diverse training data and more stable fold estimates.
2. **Channel count** — 32 vs 30 channels; minor but REVE's attention mechanism benefits from more spatial context.
3. **Signal quality** — THU-EP binary AUROC variance is ±5.01% vs ±2.58% for FACED, suggesting higher trial-to-trial or subject-to-subject noise.
4. **Data exclusions** — THU-EP excludes one full subject and partial stimuli for two others, introducing minor imbalances.

### 4.2 Binary vs 9-class

| Metric | FACED Binary | FACED 9-class | THU-EP Binary | THU-EP 9-class |
|---|---|---|---|---|
| Acc above chance (pp) | +20.57 | +37.81 | +15.07 | +29.96 |
| AUROC above chance | +26.83 | +34.16 | +21.32 | +29.99 |

When expressed as margin above chance, 9-class shows a *larger* absolute margin than binary. This is counterintuitive but simply reflects the much lower chance floor (11.1% vs 50%). In relative terms, binary still dominates: FACED binary is 1.41× chance vs 4.41× for 9-class, but the 9-class floor is so low that even moderate learning looks large in absolute pp.

### 4.3 The AUROC–Accuracy Dissociation

| Experiment | Accuracy | AUROC | Gap |
|---|---|---|---|
| FACED 9-class (CV) | 48.91% | 84.16% | 35.25 pp |
| THU-EP 9-class (CV) | 41.06% | 79.99% | 38.93 pp |
| FACED 9-class (Gen) | 15.83% | ~53–54% | ~37 pp |
| THU-EP 9-class (Gen) | 17.50% | ~56% | ~39 pp |

High AUROC with moderate accuracy is diagnostic:

- The model **ranks the correct class highly** even when argmax is wrong. In 9-class, being the 2nd most probable class counts as an error in accuracy but not in AUROC.
- **Adjacent emotions are genuinely confusable** in EEG (e.g. Anger vs Fear, Joy vs Amusement). The model learns the right emotion neighbourhood but struggles with fine boundaries.
- The **linear head is the bottleneck** — the representations already contain the necessary information (high AUROC), but a linear boundary cannot exploit it fully. Fine-tuning the encoder or using a non-linear head should close this gap.

### 4.4 CV vs Stimulus Generalisation Gap

| Dataset | Task | CV Acc | Gen Acc | Drop | % of CV retained |
|---|---|---|---|---|---|
| FACED | Binary | 70.57% | 58.46% | −12.11 pp | 82.9% |
| FACED | 9-class | 48.91% | 15.83% | −33.08 pp | 32.4% |
| THU-EP | Binary | 65.07% | 55.59% | −9.48 pp | 85.4% |
| THU-EP | 9-class | 41.06% | 17.50% | −23.56 pp | 42.6% |

Binary tasks retain ~83–85% of CV performance when stimuli change. 9-class tasks collapse to only 32–43% of CV performance.

- **Valence (binary) partially generalises** — there is real stimulus-independent emotional valence signal in the REVE representations, though with a ~10 pp accuracy penalty.
- **Fine-grained emotion (9-class) is largely stimulus-bound** — the model relies heavily on which specific stimulus was shown. EEG patterns at fine emotional granularity mix genuine emotion with perceptual and cognitive responses specific to the stimulus content, and linear probing cannot disentangle them.

---

## 5. Summary and Implications

### Performance at a Glance

| Condition | Dataset | Task | Accuracy | Reading |
|---|---|---|---|---|
| Cross-subject CV | FACED | Binary | **70.6%** | Strong — well above chance, consistent |
| Cross-subject CV | FACED | 9-class | **48.9%** | Solid — 4.4× chance, excellent AUROC (84%) |
| Cross-subject CV | THU-EP | Binary | **65.1%** | Good — above chance, higher variance |
| Cross-subject CV | THU-EP | 9-class | **41.1%** | Moderate — 3.7× chance, good AUROC (80%) |
| Stim. gen. | FACED | Binary | **58.5%** | Modest — above chance but meaningful drop |
| Stim. gen. | FACED | 9-class | **15.8%** | Near-chance — 1.4× chance only |
| Stim. gen. | THU-EP | Binary | **55.6%** | Modest — just above chance |
| Stim. gen. | THU-EP | 9-class | **17.5%** | Near-chance — 1.6× chance only |

### Key Takeaways

1. **REVE linear probing works.** All four CV settings show genuine learning well above chance.
2. **Valence (binary) is the robust signal.** Cross-subject binary accuracy of 65–71%, retaining ~84% of that under stimulus generalisation.
3. **Fine-grained emotion (9-class) is largely stimulus-bound.** CV looks impressive (41–49%) but generalisation collapses to near chance — the model is learning stimulus-correlated patterns, not pure emotional states.
4. **The encoder ranks correctly; the linear head is the bottleneck.** The AUROC–accuracy gap (up to 39 pp) in 9-class CV means the representations are already good — fine-tuning the encoder or using a non-linear head should convert ranking ability into accuracy.
5. **THU-EP is harder and noisier.** Fewer subjects and higher variance throughout; improvements on FACED should transfer, but with a persistent gap.
6. **Training budget is too tight.** Many folds peaked at epoch 20 (the maximum). Extending to 30–50 epochs is likely to yield free gains.

### Implications for Future Work

| Future direction | What the current results suggest |
|---|---|
| **LoRA fine-tuning** | High AUROC but moderate 9-class accuracy means the encoder's geometry is already useful — fine-tuning should sharpen boundaries and boost accuracy directly. |
| **Supervised Contrastive Learning** | The generalisation collapse implies representations cluster by stimulus, not just by emotion. SupCon with emotion-level positives/negatives would push same-emotion, different-stimulus windows together, directly attacking the stimulus confound. |
| **Non-linear classification head** | AUROC >> accuracy in 9-class indicates the linear boundary is suboptimal; even a small MLP head could better exploit the existing representation geometry. |
| **Longer training** | Many folds still improving at epoch 20 — extend `max_epochs` to 30–50 with current early stopping patience. |
| **More seeds in generalisation** | Binary shows 2.1% seed variance; 5+ seeds would give a more stable estimate of generalisation performance. |
