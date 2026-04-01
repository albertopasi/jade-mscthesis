# Fine-Tuning Results Analysis

**Experiment:** REVE fine-tuning (LoRA and full fine-tuning) on FACED and THU-EP EEG emotion datasets

**Architecture:** Two-stage pipeline — LP warmup (frozen encoder, trainable `cls_query_token` + linear head), followed by LoRA fine-tuning or full fine-tuning of the encoder.

**Window:** 10 s (2000 pts at 200 Hz), non-overlapping, `pooling=no`

**Completed:** 2026-04-01

---

## 1. Experimental Conditions

| Axis | Values |
|---|---|
| **Dataset** | FACED (123 subjects), THU-EP (79 subjects) |
| **Task** | 9-class emotion, binary (positive vs. negative) |
| **Method** | LoRA (rank=16, `ft_lora`), Full fine-tuning (`ft_fullft`) |
| **Augmentation** | Mixup enabled vs. disabled (`nomixup`) |
| **Pooling** | `no` (full query+patch concatenation), `last` (query token only) |
| **Evaluation** | 10-fold cross-subject CV, static split (`revesplit`), stimulus generalisation |

### LP Warmup Reference

The LP warmup stage at the start of each FT run is identical in configuration to the standalone LP experiment (frozen encoder, 20 epochs max, patience=10, StableAdamW, LR=5e-3). LP warmup accuracy is recorded per fold and serves as a baseline for measuring FT improvement.

---

## 2. Metrics Glossary

| Metric | Description |
|---|---|
| **val_acc** | Fraction of windows correctly classified on held-out subjects |
| **val_bal_acc** | Mean per-class recall; corrects for label imbalance |
| **val_auroc** | Macro one-vs-rest AUC; measures probability ranking quality |
| **val_f1** | Weighted F1 score |
| **train_loss** | Final training loss at checkpoint (available for full-FT runs only) |
| **val_loss** | Validation loss at best checkpoint (full-FT runs only) |
| **lp_val_acc** | Validation accuracy at end of LP warmup stage (before FT begins) |

---

## 3. Main Results: Cross-Subject 10-Fold CV

### 3.1 Summary Table — LoRA (rank=16, pooling=no)

All results are mean ± std across 10 folds.

| Dataset | Task | Mixup | val_acc | val_bal_acc | val_auroc | val_f1 | Avg FT epochs |
|---|---|---|---|---|---|---|---|
| FACED | 9-class | Yes | **0.5042 ± 0.0369** | 0.5053 ± 0.0385 | 0.8409 ± 0.0312 | 0.5037 ± 0.0370 | ~20 |
| FACED | 9-class | No | **0.5362 ± 0.0371** | 0.5373 ± 0.0383 | 0.8561 ± 0.0268 | 0.5361 ± 0.0368 | ~28 |
| FACED | binary | Yes | **0.7027 ± 0.0217** | 0.7027 ± 0.0217 | 0.7577 ± 0.0318 | 0.7023 ± 0.0215 | ~19 |
| FACED | binary | No | **0.7250 ± 0.0163** | 0.7250 ± 0.0163 | 0.7843 ± 0.0232 | 0.7237 ± 0.0171 | ~22 |
| THU-EP | 9-class | Yes | **0.4357 ± 0.0237** | 0.4364 ± 0.0230 | 0.8132 ± 0.0196 | 0.4326 ± 0.0246 | ~22 |
| THU-EP | 9-class | No | **0.4668 ± 0.0215** | 0.4666 ± 0.0235 | 0.8304 ± 0.0157 | 0.4637 ± 0.0226 | ~30 |
| THU-EP | binary | Yes | **0.6700 ± 0.0270** | 0.6700 ± 0.0270 | 0.7133 ± 0.0368 | 0.6684 ± 0.0275 | ~25 |
| THU-EP | binary | No | **0.6827 ± 0.0249** | 0.6827 ± 0.0249 | 0.7353 ± 0.0346 | 0.6804 ± 0.0260 | ~25 |

### 3.2 Summary Table — Full Fine-Tuning (pooling=no, FACED only)

10-fold cross-subject CV results (no static split).

| Dataset | Task | Mixup | val_acc | val_bal_acc | val_auroc | val_f1 | Mean train_loss | Avg FT epochs |
|---|---|---|---|---|---|---|---|---|
| FACED | 9-class | Yes | **0.5490 ± 0.0370** | 0.5503 ± 0.0373 | 0.8669 ± 0.0289 | 0.5483 ± 0.0362 | 0.8119 ± 0.1681 | ~65 |
| FACED | 9-class | No | **0.5820 ± 0.0332** | 0.5844 ± 0.0355 | 0.8836 ± 0.0271 | 0.5820 ± 0.0323 | 0.0012 ± 0.0021 | ~82 |
| FACED | binary | Yes | **0.7402 ± 0.0191** | 0.7402 ± 0.0191 | 0.8016 ± 0.0278 | 0.7393 ± 0.0187 | 0.5525 ± 0.1629 | ~60 |
| FACED | binary | No | **0.7555 ± 0.0216** | 0.7555 ± 0.0216 | 0.8230 ± 0.0241 | 0.7545 ± 0.0223 | 0.3738 ± 0.5030 | ~54 |

### 3.3 LoRA with `pooling=last` (no-mixup only)

Tested as an alternative to concatenation pooling.

| Dataset | Task | Pooling | val_acc | val_bal_acc | val_auroc | val_f1 |
|---|---|---|---|---|---|---|
| FACED | binary | last | **0.7364 ± 0.0230** | 0.7364 ± 0.0230 | 0.7992 ± 0.0321 | 0.7360 ± 0.0233 |
| THU-EP | binary | last | **0.6363 ± 0.0170** | 0.6362 ± 0.0169 | 0.6740 ± 0.0232 | 0.6339 ± 0.0187 |
| THU-EP | 9-class | last | **0.3805 ± 0.0404** | 0.3785 ± 0.0414 | 0.7760 ± 0.0357 | 0.3752 ± 0.0408 |

---

## 4. Static Split Results (FACED, `--revesplit`)

Train: subjects 0–79 (80 subjects), Val: 80–99 (20 subjects), Test: 100–122 (23 subjects).

| Task | Mixup | Val acc | Val AUROC | Test acc | Test AUROC | FT epochs |
|---|---|---|---|---|---|---|
| 9-class | Yes | 0.4720 | 0.8213 | **0.4099** | 0.7969 | 65 |
| 9-class | No | 0.4881 | 0.8312 | **0.4596** | 0.8269 | 73 |
| binary | Yes | 0.6813 | 0.7328 | **0.6763** | 0.7449 | 48 |
| binary | No | 0.6979 | 0.7522 | **0.7089** | 0.7777 | 55 |

Key observation: test accuracy drops notably from val accuracy in the 9-class task (e.g. 0.4881 val → 0.4596 test for no-mixup), suggesting the static subject split does not distribute subject heterogeneity uniformly, and the test set (subjects 100–122) is harder than the validation set (subjects 80–99). The binary task is more stable across the split.

---

## 5. Stimulus Generalisation Results

Evaluation protocol: train on 2/3 of stimuli per emotion × train subjects; validate on held-out 1/3 of stimuli × val subjects. Reported as cross-seed mean ± std over seeds {123, 456, 789}.

### 5.1 FACED — Binary, LoRA

| Seed | val_acc | val_bal_acc | val_auroc | val_f1 |
|---|---|---|---|---|
| 123 | 0.5911 | 0.5911 | 0.6098 | 0.5860 |
| 456 | 0.6020 | 0.6020 | 0.6217 | 0.5970 |
| 789 | 0.5719 | 0.5719 | 0.5731 | 0.5584 |
| **Cross-seed mean ± std** | **0.5883 ± 0.0152** | **0.5883 ± 0.0152** | 0.6015 | **0.5805 ± 0.0199** |

### 5.2 THU-EP — 9-class, LoRA

| Seed | val_acc | val_bal_acc | val_auroc | val_f1 |
|---|---|---|---|---|
| 123 | 0.1712 | 0.1713 | 0.5504 | 0.1543 |
| 456 | 0.1678 | 0.1676 | 0.5560 | 0.1435 |
| 789 | 0.1704 | 0.1706 | 0.5567 | 0.1515 |
| **Cross-seed mean ± std** | **0.1698 ± 0.0018** | **0.1698 ± 0.0020** | 0.5544 | **0.1498 ± 0.0056** |

The random-chance baseline for 9-class is 1/9 ≈ 0.111; these results are above chance but only marginally so. The AUROC of ~0.55 is barely above chance (0.50), indicating that neither cross-subject transfer nor cross-stimulus generalisation is reliably achieved for the THU-EP 9-class task under stimulus generalisation conditions.

---

## 6. Comparison with Linear Probing Baseline

LP reference results (official mode, pooling=no, 10-fold cross-subject):

| Dataset | Task | LP val_acc | LP val_auroc |
|---|---|---|---|
| FACED | 9-class | 0.4891 ± 0.0286 | 0.8416 ± 0.0279 |
| FACED | binary | 0.7057 ± 0.0156 | 0.7683 ± 0.0258 |
| THU-EP | 9-class | 0.4106 ± 0.0206 | 0.7999 ± 0.0231 |
| THU-EP | binary | 0.6507 ± 0.0331 | 0.7132 ± 0.0501 |

### Improvement over LP (LoRA, no-mixup)

| Dataset | Task | LP val_acc | FT-LoRA val_acc | Δ acc | FT-LoRA val_auroc | Δ AUROC |
|---|---|---|---|---|---|---|
| FACED | 9-class | 0.4891 | 0.5362 | **+0.0471** | 0.8561 | +0.0145 |
| FACED | binary | 0.7057 | 0.7250 | **+0.0193** | 0.7843 | +0.0160 |
| THU-EP | 9-class | 0.4106 | 0.4668 | **+0.0562** | 0.8304 | +0.0305 |
| THU-EP | binary | 0.6507 | 0.6827 | **+0.0320** | 0.7353 | +0.0221 |

### Improvement over LP (Full FT, no-mixup, FACED)

| Dataset | Task | LP val_acc | Full-FT val_acc | Δ acc |
|---|---|---|---|---|
| FACED | 9-class | 0.4891 | 0.5820 | **+0.0929** |
| FACED | binary | 0.7057 | 0.7555 | **+0.0498** |

Fine-tuning consistently improves over LP. Full FT provides larger gains than LoRA, particularly for the 9-class task (+9.3 pp vs +4.7 pp for LoRA).

---

## 7. Analysis

### 7.1 LoRA vs. Full Fine-Tuning

LoRA fine-tuning (rank=16) and full fine-tuning both improve substantially over LP, but full FT achieves higher accuracy in all tested conditions on FACED:

| Task | Mixup | LoRA val_acc | Full FT val_acc | Δ |
|---|---|---|---|---|
| FACED 9-class | Yes | 0.5042 | 0.5490 | +0.0448 |
| FACED 9-class | No | 0.5362 | 0.5820 | +0.0458 |
| FACED binary | Yes | 0.7027 | 0.7402 | +0.0375 |
| FACED binary | No | 0.7250 | 0.7555 | +0.0305 |

Full FT provides a consistent 3–5 pp gain over LoRA. This is expected: full FT unfreezes all encoder parameters, giving the model complete freedom to adapt its representations to the target domain. LoRA constrains the update to low-rank subspaces (rank=16 out of 512-dimensional projections), which limits representational plasticity but drastically reduces the risk of catastrophic forgetting and the number of trainable parameters.

However, the full FT results come at a significant cost: training is considerably longer (mean FT epochs ~54–82 vs ~20–30 for LoRA), and the train loss in the full-FT no-mixup regime approaches zero (mean 0.0012 for 9-class), strongly indicating that the model is memorising the training set. The very high val_loss values observed for full FT (e.g. fold 9: val_loss = 17.12 for 9-class no-mixup; fold 3: val_loss = 10.17) confirm severe overfitting in some folds, despite early stopping. Validation accuracy is still better than LoRA because the model has more total capacity and the early-stopped checkpoint captures a genuinely better generalising state, but the training dynamics are unstable.

For LoRA, training losses are not reported directly in these outputs, but the shorter FT epoch counts and consistent performance across folds indicate smoother, more stable optimisation. The LoRA LP warmup training losses (where available) are on the order of 1–2, which is appropriate for 9-class cross-entropy.

**Conclusion:** LoRA is the safer default for this problem. Full FT yields higher peak accuracy but requires careful regularisation and exhibits instability. For production use or extended experiments, LoRA with augmentation (mixup) is the recommended configuration.

### 7.2 Effect of Mixup Augmentation

Mixup consistently degrades performance relative to no-mixup across all datasets, tasks, and approaches:

| Dataset | Task | Method | Mixup val_acc | No-mixup val_acc | Δ (no-mixup − mixup) |
|---|---|---|---|---|---|
| FACED | 9-class | LoRA | 0.5042 | 0.5362 | **+0.0320** |
| FACED | binary | LoRA | 0.7027 | 0.7250 | **+0.0223** |
| THU-EP | 9-class | LoRA | 0.4357 | 0.4668 | **+0.0311** |
| THU-EP | binary | LoRA | 0.6700 | 0.6827 | **+0.0127** |
| FACED | 9-class | Full FT | 0.5490 | 0.5820 | **+0.0330** |
| FACED | binary | Full FT | 0.7402 | 0.7555 | **+0.0153** |

The no-mixup configuration uniformly outperforms mixup by 1–3 pp in accuracy and shows higher AUROC across the board. This is a notable reversal compared to the LP experiment, where mixup hurt performance by a smaller and less consistent margin (LP FACED 9-class: mixup 0.4891 vs no-mixup 0.4946, Δ+0.0055).

Several factors explain the larger harmful effect of mixup in FT:

1. **Curriculum disruption.** The LP warmup initialises the head and query token well. Mixup distorts the loss signal during FT, making the warmup initialisation harder to exploit — the model must recover from blended training targets before learning meaningful class-specific representations.

2. **Convex combination mismatch.** Mixup creates interpolated samples that do not correspond to real EEG signals. The REVE encoder, pretrained on real neural data, produces embeddings that may not lie on the class-separating manifold for synthetic blends. This forces the model to waste capacity on artefact-induced gradients.

3. **No-mixup trains longer.** No-mixup runs converge at later epochs (e.g. LoRA FACED 9-class: mean ~28 FT epochs without mixup vs ~20 with mixup), consistent with better fine-grained optimisation. With mixup, early stopping triggers earlier because the loss landscape is noisier.

4. **Full FT no-mixup train_loss ≈ 0** is evidence that without augmentation the model overfits the train set — yet val accuracy is higher. This suggests that for the full FT setting, the early-stopping checkpoint is consistently selected before catastrophic overfit sets in, and that the additional capacity of full FT is enough to learn genuinely better representations even without regularisation.

**Recommendation:** For all subsequent FT experiments, disable mixup. The regularisation benefit of mixup is outweighed by the disruption it causes to the two-stage pipeline.

### 7.3 Pooling Strategy: `no` vs. `last`

The `no` pooling variant concatenates the query token and all patch embeddings into a very high-dimensional vector (FACED 32 channels × 11 patches + 1 query × 512 = 180,736 dims), while `last` uses only the scalar query token output (512 dims).

Comparison (LoRA, no-mixup):

| Dataset | Task | Pooling `no` val_acc | Pooling `last` val_acc | Δ |
|---|---|---|---|---|
| FACED | binary | 0.7250 | 0.7364 | **+0.0114** (last) |
| THU-EP | binary | 0.6827 | 0.6363 | **−0.0464** (no better) |
| THU-EP | 9-class | 0.4668 | 0.3805 | **−0.0863** (no better) |

Results are mixed. `last` pooling achieves marginally higher accuracy for FACED binary (+1.1 pp), but degrades substantially for THU-EP 9-class (−8.6 pp) and binary (−4.6 pp). The query token provides a compressed, channel-agnostic summary that may suffice for the relatively balanced FACED binary task, but is insufficient to retain the channel-specific spatial information needed for fine-grained emotion classification on THU-EP.

The `no` pooling variant is more robust across datasets and tasks and should be the default.

### 7.4 Cross-Dataset Performance Gap

FACED consistently outperforms THU-EP across all configurations:

| Task | FACED (LoRA, no-mixup) | THU-EP (LoRA, no-mixup) | Gap |
|---|---|---|---|
| 9-class | 0.5362 | 0.4668 | −0.0694 |
| binary | 0.7250 | 0.6827 | −0.0423 |

The gap is consistent with findings from LP. Likely causes include:
- **Dataset size:** FACED has 123 subjects vs. 79 for THU-EP (subject 75 excluded), providing more cross-subject diversity.
- **Excluded stimuli:** Subject 37 (3 excluded stimuli) and subject 46 (5 excluded stimuli) reduce effective sample size unevenly across folds.
- **Recording conditions:** THU-EP data may be noisier or more heterogeneous across sessions.
- **9-class random chance:** 1/9 ≈ 11.1%. THU-EP at 46.7% is still a strong result given the task complexity, but the ceiling is simply lower than FACED.

### 7.5 Within-Fold Variance and Stability

Fold-level inspection reveals substantial within-experiment variance for both datasets:

**FACED 9-class LoRA no-mixup — fold range:**
- Worst fold: fold 3 — 0.4679 (AUROC 0.8064)
- Best fold: fold 7 — 0.5823 (AUROC 0.8782)
- Spread: ~11.4 pp

**THU-EP 9-class LoRA no-mixup — fold range:**
- Worst fold: fold 7 — 0.4435 (AUROC 0.8230)
- Best fold: fold 10 — 0.4949 (AUROC 0.8445)
- Spread: ~5.1 pp

**FACED full FT 9-class no-mixup — fold range:**
- Worst fold: fold 3 — 0.5137 (val_loss 10.17, extreme overfit)
- Best fold: fold 1 — 0.6154 (val_loss 6.93)
- Spread: ~10.2 pp

Variance is higher for full FT, particularly in val_loss (fold 9: 17.12), confirming instability in some folds. LoRA results are more predictable fold-to-fold.

### 7.6 AUROC vs. Accuracy: Dissociation

Across all configurations, AUROC values are substantially higher than what accuracy alone would suggest:
- FACED 9-class LoRA no-mixup: val_acc = 0.5362, val_auroc = 0.8561
- THU-EP 9-class LoRA no-mixup: val_acc = 0.4668, val_auroc = 0.8304

An AUROC of 0.83–0.88 indicates that the model produces well-calibrated probability rankings across all nine classes, even when the final hard classification is incorrect. This is characteristic of tasks with fine-grained inter-class boundaries: the model may correctly order Anger > Disgust > Fear in probability space without committing to the right peak class. For EEG emotion recognition, this dissociation between accuracy and AUROC is expected, as within-category emotion confusion is biologically plausible.

### 7.7 Two-Stage Training: LP Warmup Effectiveness

The LP warmup stage successfully initialises the head and query token before FT. Comparing lp_val_acc (end of LP warmup) to final FT val_acc:

**FACED 9-class LoRA no-mixup (selected folds):**
| Fold | lp_val_acc | final val_acc | FT gain |
|---|---|---|---|
| 1 | 0.5082 | 0.5549 | +0.0467 |
| 2 | 0.4936 | 0.5293 | +0.0357 |
| 7 | 0.5516 | 0.5823 | +0.0307 |
| 9 | 0.4901 | 0.5000 | +0.0099 |

FT reliably improves over LP warmup by 3–5 pp in most folds. The smallest gains (fold 9 above: +0.01) occur when the LP warmup itself struggles, suggesting some folds have harder subject splits that limit how much FT can recover.

For full FT (FACED binary no-mixup), LP warmup converges to `lp_val_acc` values in the range 0.64–0.73, and FT final accuracies reach 0.73–0.78 — a consistent improvement of ~3–5 pp even from a stronger LP starting point.

---

## 8. Stimulus Generalisation: Analysis

### 8.1 FACED Binary

Cross-seed accuracy of 0.5883 ± 0.0152 (random chance = 0.50 for binary). This represents a meaningful but modest generalisation: the model transfers some emotional valence signal to unseen subjects and unseen stimuli, but the double-hold-out dramatically reduces accuracy relative to cross-subject CV (0.5883 vs 0.7250 for standard CV). The AUROC of ~0.60 is well above chance (0.50) but modest in absolute terms.

Seed-level variance (std = 0.0152) is small, indicating the result is not a statistical artefact of the particular stimulus split chosen.

### 8.2 THU-EP 9-Class

Cross-seed accuracy of 0.1698 ± 0.0018 (chance = 0.111). At first glance, the model is above chance, but the AUROC of ~0.554 is only marginally above 0.50. The F1 of 0.1498 is very low, indicating the model essentially defaults to predicting the majority class for most subjects under this condition.

The contrast with the standard cross-subject CV (0.4668) is striking: stimulus generalisation reduces accuracy by ~27 pp and nearly collapses AUROC from 0.83 to 0.55. This collapse indicates that:
1. The REVE-based features learn stimulus-specific rather than truly emotion-generic representations for THU-EP.
2. Different stimuli evoking the same nominal emotion elicit sufficiently different EEG patterns that the model cannot generalise the label from one set of stimuli to another.
3. The smaller THU-EP dataset (fewer subjects, several excluded stimuli) compounds this problem.

### 8.3 Interpretation

Stimulus generalisation is a strictly harder task than cross-subject generalisation because it simultaneously requires two types of transfer: to unseen people **and** to unseen stimuli. The gap between standard CV and generalisation results is evidence that current REVE fine-tuning captures subject-invariant emotion signal at the level of acoustic/perceptual features of the specific stimuli seen in training, rather than abstract emotional state.

---

## 9. Static Split Results: Analysis

For the FACED `--revesplit` condition, full FT test accuracy on the 9-class task:
- With mixup: **0.4099** (val: 0.4720, gap = 6.2 pp)
- Without mixup: **0.4596** (val: 0.4881, gap = 2.9 pp)

The val→test gap is substantially larger with mixup, suggesting that mixup causes the model to overfit the subject distribution in the training set in a way that exacerbates mismatch against the test subjects (100–122). Without mixup, the test set generalises more faithfully from the val set.

For binary:
- With mixup: test = **0.6763** (val = 0.6813, gap = 0.5 pp)
- Without mixup: test = **0.7089** (val = 0.6979, gap = −1.1 pp; test slightly *higher* than val)

The binary task shows negligible val→test gap and stable generalisation. The fact that no-mixup test accuracy slightly exceeds val accuracy is consistent with random variation under the small test set (23 subjects).

---

## 10. Summary of Best Configurations

### 10.1 Overall Best per Setting

| Scenario | Best config | val_acc |
|---|---|---|
| FACED 9-class (cross-subject CV) | Full FT, no-mixup, pooling=no | **0.5820 ± 0.0332** |
| FACED binary (cross-subject CV) | Full FT, no-mixup, pooling=no | **0.7555 ± 0.0216** |
| THU-EP 9-class (cross-subject CV) | LoRA, no-mixup, pooling=no | **0.4668 ± 0.0215** |
| THU-EP binary (cross-subject CV) | LoRA, no-mixup, pooling=no | **0.6827 ± 0.0249** |
| FACED 9-class (static split, test) | Full FT, no-mixup | **0.4596** |
| FACED binary (static split, test) | Full FT, no-mixup | **0.7089** |
| FACED binary (stimulus gen.) | LoRA, mixup | **0.5883 ± 0.0152** |
| THU-EP 9-class (stimulus gen.) | LoRA, mixup | **0.1698 ± 0.0018** |

### 10.2 Ranking by Approach (FACED only, 9-class)

Ordered by val_acc descending:

| Rank | Method | Mixup | val_acc |
|---|---|---|---|
| 1 | Full FT | No | **0.5820** |
| 2 | Full FT | Yes | 0.5490 |
| 3 | LoRA | No | 0.5362 |
| 4 | LoRA | Yes | 0.5042 |
| 5 | LP (baseline) | Yes | 0.4891 |

---

## 11. Overall Conclusions

1. **Fine-tuning consistently improves over linear probing.** Across all datasets and tasks, LoRA FT and full FT achieve higher accuracy, balanced accuracy, and AUROC than LP alone. The improvement is most pronounced for the 9-class task (+4.7 pp for LoRA, +9.3 pp for full FT on FACED).

2. **Full FT outperforms LoRA, but at the cost of stability.** Full FT achieves 3–5 pp higher accuracy across tasks, but exhibits near-zero training loss and very high validation loss in several folds under the no-mixup regime, indicating memorisation. Early stopping prevents catastrophic collapse, but training dynamics are unstable. LoRA is the safer, more reproducible approach.

3. **Mixup hurts fine-tuning performance uniformly.** No-mixup outperforms mixup by 1–3 pp across all methods, datasets, and tasks. This is contrary to its typical role as a beneficial regulariser in supervised learning, and likely reflects a mismatch between synthetic blended inputs and the distribution of embeddings produced by a frozen or partially frozen pretrained encoder. Future experiments should default to no-mixup for FT.

4. **Pooling `no` (full concatenation) is generally better than `last` (query token only).** The exception is FACED binary where `last` provides a marginal +1.1 pp advantage, but for THU-EP and for 9-class tasks, `no` pooling is substantially better. The full spatial-temporal embedding carries useful information beyond the compressed query token.

5. **Stimulus generalisation reveals limited cross-stimulus transfer.** Under the double hold-out (unseen subjects + unseen stimuli), FACED binary performance degrades to ~59% (vs 73% for standard CV), and THU-EP 9-class collapses to near-chance. Fine-tuned REVE representations appear to capture stimulus-specific rather than truly emotion-generic signals, at least for THU-EP.

6. **AUROC substantially exceeds accuracy**, particularly for 9-class tasks (AUROC ~0.83–0.88 vs. accuracy ~0.47–0.58). This indicates that the model assigns well-calibrated probability scores even when hard decisions are wrong — the model "knows" the right emotion is one of the top candidates but may not resolve fine-grained intra-category boundaries.

7. **THU-EP remains harder than FACED** across all conditions, with 6–9 pp lower accuracy for 9-class and 4–5 pp lower for binary classification. This is consistent with the smaller effective training set size (79 vs. 123 subjects) and potential recording heterogeneity in THU-EP.
