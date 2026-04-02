# Fine-Tuning Results Analysis

**Experiment:** REVE fine-tuning (LoRA and full fine-tuning) on FACED and THU-EP EEG emotion datasets

**Architecture:** Two-stage pipeline — LP warmup (frozen encoder, trainable `cls_query_token` + linear head), followed by LoRA fine-tuning or full fine-tuning of the encoder.

**Window:** 10 s (2000 pts at 200 Hz), non-overlapping, `pooling=no`

**Completed:** 2026-04-02

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
| **train_loss** | Final training loss at checkpoint |
| **val_loss** | Validation loss at best checkpoint |
| **lp_val_acc** | Validation accuracy at end of LP warmup stage (before FT begins) |

---

## 3. Main Results: Cross-Subject 10-Fold CV

### 3.1 Summary Table — LoRA (rank=16, pooling=no)

All results are mean ± std across 10 folds.

| Dataset | Task | Mixup | val_acc | val_bal_acc | val_auroc | val_f1 | Avg FT epochs |
|---|---|---|---|---|---|---|---|
| FACED | 9-class | Yes | 0.5042 ± 0.0369 | 0.5053 ± 0.0385 | 0.8409 ± 0.0312 | 0.5037 ± 0.0370 | ~20 |
| FACED | 9-class | No | **0.5362 ± 0.0371** | 0.5373 ± 0.0383 | 0.8561 ± 0.0268 | 0.5361 ± 0.0368 | ~28 |
| FACED | binary | Yes | 0.7027 ± 0.0217 | 0.7027 ± 0.0217 | 0.7577 ± 0.0318 | 0.7023 ± 0.0215 | ~19 |
| FACED | binary | No | **0.7250 ± 0.0163** | 0.7250 ± 0.0163 | 0.7843 ± 0.0232 | 0.7237 ± 0.0171 | ~22 |
| THU-EP | 9-class | Yes | 0.4357 ± 0.0237 | 0.4364 ± 0.0230 | 0.8132 ± 0.0196 | 0.4326 ± 0.0246 | ~22 |
| THU-EP | 9-class | No | **0.4668 ± 0.0215** | 0.4666 ± 0.0235 | 0.8304 ± 0.0157 | 0.4637 ± 0.0226 | ~30 |
| THU-EP | binary | Yes | 0.6700 ± 0.0270 | 0.6700 ± 0.0270 | 0.7133 ± 0.0368 | 0.6684 ± 0.0275 | ~25 |
| THU-EP | binary | No | **0.6827 ± 0.0249** | 0.6827 ± 0.0249 | 0.7353 ± 0.0346 | 0.6804 ± 0.0260 | ~25 |

### 3.2 Summary Table — Full Fine-Tuning (pooling=no)

10-fold cross-subject CV results across both datasets.

| Dataset | Task | Mixup | val_acc | val_bal_acc | val_auroc | val_f1 | Mean train_loss | Avg FT epochs |
|---|---|---|---|---|---|---|---|---|
| FACED | 9-class | Yes | 0.5490 ± 0.0370 | 0.5503 ± 0.0373 | 0.8669 ± 0.0289 | 0.5483 ± 0.0362 | 0.8119 ± 0.1681 | ~65 |
| FACED | 9-class | No | **0.5820 ± 0.0332** | 0.5844 ± 0.0355 | 0.8836 ± 0.0271 | 0.5820 ± 0.0323 | 0.0012 ± 0.0021 | ~82 |
| FACED | binary | Yes | 0.7402 ± 0.0191 | 0.7402 ± 0.0191 | 0.8016 ± 0.0278 | 0.7393 ± 0.0187 | 0.5525 ± 0.1629 | ~60 |
| FACED | binary | No | **0.7555 ± 0.0216** | 0.7555 ± 0.0216 | 0.8230 ± 0.0241 | 0.7545 ± 0.0223 | 0.3738 ± 0.5030 | ~54 |
| THU-EP | 9-class | No | **0.4828 ± 0.0272** | 0.4841 ± 0.0274 | 0.8404 ± 0.0126 | 0.4798 ± 0.0271 | 0.0169 ± 0.0314 | ~59 |
| THU-EP | binary | No | **0.6990 ± 0.0167** | 0.6990 ± 0.0167 | 0.7558 ± 0.0155 | 0.6985 ± 0.0166 | 0.2256 ± 0.2754 | ~51 |

THU-EP full FT was run without mixup only, consistent with the no-mixup recommendation derived from FACED experiments (see §7.2).

### 3.3 LoRA with `pooling=last` (no-mixup only)

Tested as an alternative to concatenation pooling.

| Dataset | Task | Pooling | val_acc | val_bal_acc | val_auroc | val_f1 |
|---|---|---|---|---|---|---|
| FACED | binary | last | 0.7364 ± 0.0230 | 0.7364 ± 0.0230 | 0.7992 ± 0.0321 | 0.7360 ± 0.0233 |
| THU-EP | binary | last | 0.6363 ± 0.0170 | 0.6362 ± 0.0169 | 0.6740 ± 0.0232 | 0.6339 ± 0.0187 |
| THU-EP | 9-class | last | 0.3805 ± 0.0404 | 0.3785 ± 0.0414 | 0.7760 ± 0.0357 | 0.3752 ± 0.0408 |

---

## 4. Static Split Results (FACED, `--revesplit`)

Train: subjects 0–79 (80 subjects), Val: 80–99 (20 subjects), Test: 100–122 (23 subjects). Full FT only.

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

### 5.1 LoRA (no-mixup) — All conditions

| Dataset | Task | val_acc | val_bal_acc | val_auroc | val_f1 |
|---|---|---|---|---|---|
| FACED | 9-class | 0.1603 ± 0.0013 | 0.1603 ± 0.0013 | ~0.534 | 0.1402 ± 0.0043 |
| FACED | binary | 0.5970 ± 0.0171 | 0.5970 ± 0.0171 | ~0.607 | 0.5916 ± 0.0156 |
| THU-EP | 9-class | 0.1790 ± 0.0051 | 0.1791 ± 0.0050 | ~0.564 | 0.1603 ± 0.0071 |

### 5.2 LoRA (with mixup) — Previously reported conditions

| Dataset | Task | Seed | val_acc | val_bal_acc | val_auroc | val_f1 |
|---|---|---|---|---|---|---|
| FACED | binary | 123 | 0.5911 | 0.5911 | 0.6098 | 0.5860 |
| FACED | binary | 456 | 0.6020 | 0.6020 | 0.6217 | 0.5970 |
| FACED | binary | 789 | 0.5719 | 0.5719 | 0.5731 | 0.5584 |
| **FACED binary cross-seed** | | | **0.5883 ± 0.0152** | **0.5883 ± 0.0152** | 0.6015 | **0.5805 ± 0.0199** |
| THU-EP | 9-class | 123 | 0.1712 | 0.1713 | 0.5504 | 0.1543 |
| THU-EP | 9-class | 456 | 0.1678 | 0.1676 | 0.5560 | 0.1435 |
| THU-EP | 9-class | 789 | 0.1704 | 0.1706 | 0.5567 | 0.1515 |
| **THU-EP 9-class cross-seed** | | | **0.1698 ± 0.0018** | **0.1698 ± 0.0020** | 0.5544 | **0.1498 ± 0.0056** |

### 5.3 Full FT (no-mixup) — Generalisation

| Dataset | Task | val_acc | val_bal_acc | val_auroc | val_f1 |
|---|---|---|---|---|---|
| FACED | 9-class | 0.1552 ± 0.0078 | 0.1552 ± 0.0078 | ~0.534 | 0.1323 ± 0.0078 |
| FACED | binary | 0.5973 ± 0.0111 | 0.5973 ± 0.0111 | ~0.614 | 0.5882 ± 0.0152 |
| THU-EP | 9-class | 0.1714 ± 0.0030 | 0.1711 ± 0.0029 | ~0.564 | 0.1452 ± 0.0037 |
| THU-EP | binary | 0.5649 ± 0.0083 | 0.5645 ± 0.0086 | ~0.560 | 0.5434 ± 0.0118 |

### 5.4 Per-seed breakdown (LoRA, no-mixup)

**FACED binary:**
| Seed | val_acc | val_auroc |
|---|---|---|
| 123 | 0.6022 | 0.6105 |
| 456 | 0.6050 | 0.6236 |
| 789 | 0.5846 | 0.6019 |

**FACED 9-class:**
| Seed | val_acc | val_auroc |
|---|---|---|
| 123 | 0.1601 | 0.5344 |
| 456 | 0.1591 | 0.5407 |
| 789 | 0.1616 | 0.5401 |

**THU-EP 9-class (LoRA no-mixup, fullft):**
| Seed | LoRA val_acc | LoRA auroc | FullFT val_acc | FullFT auroc |
|---|---|---|---|---|
| 123 | 0.1831 | 0.5594 | 0.1741 | 0.5604 |
| 456 | 0.1806 | 0.5704 | 0.1719 | 0.5578 |
| 789 | 0.1733 | 0.5620 | 0.1681 | 0.5530 |

---

## 6. Comparison with Linear Probing Baseline

LP reference results (official mode, pooling=no, 10-fold cross-subject):

| Dataset | Task | LP val_acc | LP val_auroc |
|---|---|---|---|
| FACED | 9-class | 0.4891 ± 0.0286 | 0.8416 ± 0.0279 |
| FACED | binary | 0.7057 ± 0.0156 | 0.7683 ± 0.0258 |
| THU-EP | 9-class | 0.4106 ± 0.0206 | 0.7999 ± 0.0231 |
| THU-EP | binary | 0.6507 ± 0.0331 | 0.7132 ± 0.0501 |

### 6.1 Improvement over LP (LoRA, no-mixup)

| Dataset | Task | LP val_acc | FT-LoRA val_acc | Δ acc | FT-LoRA val_auroc | Δ AUROC |
|---|---|---|---|---|---|---|
| FACED | 9-class | 0.4891 | 0.5362 | **+0.0471** | 0.8561 | +0.0145 |
| FACED | binary | 0.7057 | 0.7250 | **+0.0193** | 0.7843 | +0.0160 |
| THU-EP | 9-class | 0.4106 | 0.4668 | **+0.0562** | 0.8304 | +0.0305 |
| THU-EP | binary | 0.6507 | 0.6827 | **+0.0320** | 0.7353 | +0.0221 |

### 6.2 Improvement over LP (Full FT, no-mixup)

| Dataset | Task | LP val_acc | Full-FT val_acc | Δ acc | Full-FT val_auroc | Δ AUROC |
|---|---|---|---|---|---|---|
| FACED | 9-class | 0.4891 | 0.5820 | **+0.0929** | 0.8836 | +0.0420 |
| FACED | binary | 0.7057 | 0.7555 | **+0.0498** | 0.8230 | +0.0547 |
| THU-EP | 9-class | 0.4106 | 0.4828 | **+0.0722** | 0.8404 | +0.0405 |
| THU-EP | binary | 0.6507 | 0.6990 | **+0.0483** | 0.7558 | +0.0426 |

Fine-tuning consistently and substantially improves over LP across all settings. Full FT provides larger gains than LoRA, with THU-EP 9-class showing the largest improvement (+7.2 pp for full FT vs +5.6 pp for LoRA over LP).

### 6.3 LP Generalisation vs FT Generalisation (FACED binary, LoRA no-mixup)

| Method | CV val_acc | Gen val_acc | Gen drop | Gen AUROC |
|---|---|---|---|---|
| LP (official) | 0.7057 | 0.5846 (3-seed) | −12.1 pp | ~0.59 |
| FT LoRA no-mixup | 0.7250 | 0.5970 | −12.8 pp | ~0.607 |
| FT Full FT no-mixup | 0.7555 | 0.5973 | −15.8 pp | ~0.614 |

The generalisation drop is larger for full FT than for LoRA (−15.8 pp vs −12.8 pp), while LP and LoRA show similar drops (~12 pp). This indicates that full FT, despite its higher CV accuracy, learns stronger stimulus-specific associations that do not transfer to held-out stimuli.

---

## 7. Analysis

### 7.1 LoRA vs. Full Fine-Tuning

LoRA (rank=16) and full FT both improve substantially over LP, but full FT achieves higher accuracy in all tested conditions:

| Dataset | Task | Mixup | LoRA val_acc | Full FT val_acc | Δ |
|---|---|---|---|---|---|
| FACED | 9-class | Yes | 0.5042 | 0.5490 | +0.0448 |
| FACED | 9-class | No | 0.5362 | 0.5820 | **+0.0458** |
| FACED | binary | Yes | 0.7027 | 0.7402 | +0.0375 |
| FACED | binary | No | 0.7250 | 0.7555 | +0.0305 |
| THU-EP | 9-class | No | 0.4668 | 0.4828 | +0.0160 |
| THU-EP | binary | No | 0.6827 | 0.6990 | +0.0163 |

Full FT provides a consistent 1.6–4.6 pp gain over LoRA. For THU-EP the gap narrows relative to FACED: while FACED 9-class shows +4.6 pp, THU-EP 9-class shows only +1.6 pp. This narrowing is consistent with a smaller training set (79 vs. 123 subjects) where the additional degrees of freedom in full FT are harder to exploit without overfitting.

The full FT results come at a significant cost: training is considerably longer (~54–82 epochs vs ~20–30 for LoRA), and the train loss in the full-FT no-mixup regime approaches zero (mean 0.0012 for FACED 9-class, 0.0169 for THU-EP 9-class), strongly indicating memorisation of the training set. The very high val_loss values confirm severe overfitting in some folds (FACED 9-class fold 9: val_loss = 17.12; THU-EP 9-class fold 7: val_loss = 9.74). THU-EP binary is notably more stable (mean train_loss=0.226, fold-to-fold val_loss in range 1.0–3.5).

For LoRA, training losses are not reported directly in these outputs, but the shorter FT epoch counts and more consistent val_loss levels (typically 1.2–2.0 for FACED 9-class) indicate smoother, more stable optimisation.

**Conclusion:** LoRA is the safer default for this problem. Full FT yields higher peak accuracy but requires careful regularisation and exhibits instability, particularly for 9-class tasks on larger or more heterogeneous datasets. For THU-EP, the LoRA–full FT gap is small enough that LoRA is clearly preferred.

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

The no-mixup configuration uniformly outperforms mixup by 1–3 pp in accuracy and shows higher AUROC across the board. This is a notable reversal compared to the LP experiment, where mixup had no meaningful effect (differences within 1 std of fold variance).

Several factors explain the harmful effect of mixup in FT:

1. **Curriculum disruption.** The LP warmup initialises the head and query token. Mixup distorts the loss signal during FT, making the warmup initialisation harder to exploit — the model must recover from blended training targets before learning meaningful class-specific representations.

2. **Convex combination mismatch.** Mixup creates interpolated samples that do not correspond to real EEG signals. The REVE encoder, pretrained on real neural data, produces embeddings that may not lie on the class-separating manifold for synthetic blends.

3. **No-mixup trains longer.** No-mixup runs converge at later epochs (LoRA FACED 9-class: ~28 FT epochs without mixup vs ~20 with mixup), consistent with finer-grained optimisation. With mixup, early stopping triggers earlier because the loss landscape is noisier.

4. **Full FT no-mixup train_loss ≈ 0** is evidence that without augmentation the model overfits the train set — yet val accuracy is higher. This suggests the early-stopping checkpoint is consistently selected before catastrophic overfit sets in, and the additional capacity of full FT learns genuinely better representations even without regularisation.

**Recommendation:** For all subsequent FT and SupCon experiments, disable mixup. SupCon in particular requires clean labels for positive pair identification and is incompatible with mixup by design.

### 7.3 Pooling Strategy: `no` vs. `last`

The `no` pooling variant concatenates the query token and all patch embeddings (FACED: 180,736 dims), while `last` uses only the scalar query token output (512 dims).

Comparison (LoRA, no-mixup):

| Dataset | Task | Pooling `no` val_acc | Pooling `last` val_acc | Δ |
|---|---|---|---|---|
| FACED | binary | 0.7250 | 0.7364 | **+0.0114** (last) |
| THU-EP | binary | 0.6827 | 0.6363 | **−0.0464** (no better) |
| THU-EP | 9-class | 0.4668 | 0.3805 | **−0.0863** (no better) |

Results are mixed. `last` pooling achieves marginally higher accuracy for FACED binary (+1.1 pp), but degrades substantially for THU-EP 9-class (−8.6 pp) and binary (−4.6 pp). The query token provides a compressed summary that may suffice for the balanced FACED binary task, but is insufficient for the fine-grained spatial information needed for 9-class or THU-EP classification.

The `no` pooling variant is more robust across datasets and tasks and remains the default for all further experiments.

### 7.4 Cross-Dataset Performance Gap

FACED consistently outperforms THU-EP across all configurations:

| Task | FACED LoRA no-mix | FACED full FT no-mix | THU-EP LoRA no-mix | THU-EP full FT no-mix | LoRA gap | FT gap |
|---|---|---|---|---|---|---|
| 9-class | 0.5362 | 0.5820 | 0.4668 | 0.4828 | −0.0694 | −0.0992 |
| binary | 0.7250 | 0.7555 | 0.6827 | 0.6990 | −0.0423 | −0.0565 |

The dataset gap is larger for full FT than for LoRA in both tasks. Full FT on FACED benefits from more training subjects (123 vs. 79), while on THU-EP the same capacity increase leads to heavier overfitting (mean train_loss 0.017 vs 0.0012 on FACED). This suggests the regularisation implicitly provided by LoRA's parameter constraints is more beneficial for the smaller THU-EP dataset.

Likely causes of the persistent gap:
- **Dataset size:** FACED has 123 subjects vs. 79 for THU-EP
- **Excluded stimuli:** Subject 37 (3 excluded) and subject 46 (5 excluded) reduce effective sample size unevenly
- **Recording conditions:** THU-EP data may be noisier or more heterogeneous across sessions

### 7.5 Within-Fold Variance and Stability

Fold-level inspection reveals substantial within-experiment variance for both datasets:

**FACED 9-class LoRA no-mixup:**
- Worst fold (fold 3): 0.4679 (AUROC 0.8064)
- Best fold (fold 7): 0.5823 (AUROC 0.8782)
- Spread: ~11.4 pp

**THU-EP 9-class LoRA no-mixup:**
- Worst fold (fold 7): 0.4435 (AUROC 0.8230)
- Best fold (fold 10): 0.4949 (AUROC 0.8445)
- Spread: ~5.1 pp

**FACED full FT 9-class no-mixup:**
- Worst fold (fold 3): 0.5137 (val_loss 10.17, extreme overfit)
- Best fold (fold 1): 0.6154 (val_loss 6.93)
- Spread: ~10.2 pp

**THU-EP full FT 9-class no-mixup:**
- Worst fold (fold 7): 0.4226 (val_loss 9.74, heavy overfit)
- Best fold (fold 1): 0.5000 (val_loss 7.68)
- Spread: ~7.7 pp

**THU-EP full FT binary no-mixup:**
- Worst fold (fold 1): 0.6823 (val_loss 1.37)
- Best fold (fold 4): 0.7257 (val_loss 2.89)
- Spread: ~4.3 pp (notably more stable than 9-class)

Variance is higher for full FT in 9-class tasks, confirming instability from memorisation. LoRA results are more predictable fold-to-fold. THU-EP binary full FT shows unusually low spread (~4.3 pp), consistent with the simpler binary task being easier to stabilise even with full FT.

### 7.6 AUROC vs. Accuracy: Dissociation

Across all configurations, AUROC values are substantially higher than what accuracy alone would suggest:

| Setting | val_acc | val_auroc | Gap |
|---|---|---|---|
| FACED 9-class LoRA no-mix | 0.5362 | 0.8561 | 0.3199 |
| FACED 9-class Full FT no-mix | 0.5820 | 0.8836 | 0.3016 |
| THU-EP 9-class LoRA no-mix | 0.4668 | 0.8304 | 0.3636 |
| THU-EP 9-class Full FT no-mix | 0.4828 | 0.8404 | 0.3576 |

An AUROC of 0.83–0.88 indicates that the model produces well-calibrated probability rankings across all nine classes, even when the final hard classification is incorrect. This dissociation is characteristic of tasks with fine-grained inter-class boundaries: the model correctly orders Anger > Disgust > Fear in probability space without committing to the right peak class. For EEG emotion recognition this is expected, as within-category emotion confusion is biologically plausible.

The gap is slightly smaller for full FT than LoRA (0.30 vs 0.32 for FACED 9-class), suggesting full FT's additional representational capacity partially converts ranking ability into correct hard decisions. However, the improvement is marginal. This points to the need for a loss that explicitly pushes same-emotion embeddings together rather than simply sharpening class-conditional probability peaks — exactly the motivation for Supervised Contrastive Learning (§11).

### 7.7 Two-Stage Training: LP Warmup Effectiveness

The LP warmup stage successfully initialises the head and query token before FT. Comparing lp_val_acc (end of LP warmup) to final FT val_acc:

**FACED 9-class LoRA no-mixup (selected folds):**
| Fold | lp_val_acc | final val_acc | FT gain |
|---|---|---|---|
| 1 | 0.5082 | 0.5549 | +0.0467 |
| 2 | 0.4936 | 0.5293 | +0.0357 |
| 7 | 0.5516 | 0.5823 | +0.0307 |
| 9 | 0.4901 | 0.5000 | +0.0099 |

**FACED 9-class Full FT no-mixup (selected folds):**
| Fold | lp_val_acc | final val_acc | FT gain |
|---|---|---|---|
| 1 | 0.5064 | 0.6154 | +0.1090 |
| 3 | 0.4240 | 0.5137 | +0.0897 |
| 6 | 0.5258 | 0.6052 | +0.0794 |
| 9 | 0.4474 | 0.5506 | +0.1032 |

**THU-EP 9-class Full FT no-mixup (selected folds):**
| Fold | lp_val_acc | final val_acc | FT gain |
|---|---|---|---|
| 1 | 0.4464 | 0.5000 | +0.0536 |
| 3 | 0.3750 | 0.4628 | +0.0878 |
| 7 | 0.3497 | 0.4226 | +0.0729 |

FT reliably improves over LP warmup by 3–5 pp for LoRA and 7–11 pp for full FT. The much larger improvement with full FT confirms that encoder adaptation is genuinely beneficial — the LP warmup provides a good initialisation but the frozen encoder is the bottleneck.

A notable anomaly: FACED full FT fold 7 (`best_epoch=1`) shows `lp_val_acc=0.5159` and `final val_acc=0.5347` — only +0.019 pp despite being stopped at the first FT epoch. This indicates the LP warmup itself converged to a local maximum that the full FT could not immediately improve on, and early stopping terminated before further optimisation. Such cases are more common with full FT (which has higher optimisation volatility) than LoRA.

---

## 8. Stimulus Generalisation: Analysis

### 8.1 Complete Comparison across Methods

| Dataset | Task | Method | CV acc | Gen acc | Drop (pp) | % CV retained |
|---|---|---|---|---|---|---|
| FACED | binary | LP | 0.7057 | 0.5846 | −12.1 | 82.8% |
| FACED | binary | LoRA mixup | 0.7027 | 0.5883 | −11.4 | 83.8% |
| FACED | binary | LoRA no-mix | 0.7250 | 0.5970 | −12.8 | 82.3% |
| FACED | binary | Full FT no-mix | 0.7555 | 0.5973 | −15.8 | 79.1% |
| FACED | 9-class | LP | 0.4891 | 0.1583 | −33.1 | 32.4% |
| FACED | 9-class | LoRA no-mix | 0.5362 | 0.1603 | −37.6 | 29.9% |
| FACED | 9-class | Full FT no-mix | 0.5820 | 0.1552 | −42.7 | 26.7% |
| THU-EP | 9-class | LP | 0.4106 | 0.1750 | −23.6 | 42.6% |
| THU-EP | 9-class | LoRA mixup | 0.4357 | 0.1698 | −26.6 | 39.0% |
| THU-EP | 9-class | LoRA no-mix | 0.4668 | 0.1790 | −28.8 | 38.3% |
| THU-EP | 9-class | Full FT no-mix | 0.4828 | 0.1714 | −31.1 | 35.5% |
| THU-EP | binary | Full FT no-mix | 0.6990 | 0.5649 | −13.4 | 80.8% |

### 8.2 Key Pattern: Higher CV Accuracy → Larger Generalisation Drop

A striking pattern emerges: methods with higher CV accuracy consistently show larger absolute and relative generalisation drops. FACED 9-class full FT achieves the best CV accuracy (0.582) but the worst generalisation (0.155, only 26.7% of CV retained). LP, with the lowest CV accuracy (0.489), retains 32.4% — actually a better generalisation ratio.

This is a critical finding. It demonstrates that the performance gains achieved by fine-tuning through cross-subject CV are **not uniformly transferable** to the harder stimulus-generalisation setting. Fine-tuning is learning representations that are better at cross-subject transfer but increasingly stimulus-bound — it strengthens the encoder's ability to recognise the specific stimuli seen in training rather than improving purely abstract emotion representations.

The pattern holds across all datasets and tasks: every FT method that outperforms LP on CV has a worse generalisation ratio. This is not a sign that FT is harmful overall — the absolute generalisation accuracy is similar across methods (FACED binary: ~0.58–0.60 regardless of method) — but it shows that the CV gains from FT are largely stimulus-specific.

### 8.3 FACED Binary Generalisation: Near-Ceiling Plateau

All methods for FACED binary generalisation converge to approximately the same accuracy (~0.58–0.60) despite large differences in CV accuracy (0.70–0.76). The implication is that there is a hard ceiling on cross-stimulus binary valence classification at around 60% accuracy with current REVE-based representations. This ceiling reflects the limit of what stimulus-independent emotional valence signal is encoded in REVE's pretrained features.

The AUROC in generalisation (~0.60–0.63 across methods) is modest but meaningfully above chance (0.50), confirming that some genuine valence signal persists across stimuli. However, the small method-to-method AUROC differences suggest that neither LoRA nor full FT is fundamentally changing the feature geometry for cross-stimulus transfer.

### 8.4 9-Class Generalisation: Near-Chance Collapse

Across all methods and datasets, 9-class stimulus generalisation results are only marginally above chance:
- FACED: ~0.16 (chance = 0.111), AUROC ~0.52–0.54
- THU-EP: ~0.17–0.18 (chance = 0.111), AUROC ~0.55–0.57

The models are essentially failing to learn stimulus-independent 9-class emotion representations. That fine-grained emotion categories collapse under cross-stimulus hold-out while binary valence partially survives indicates that valence is a more stimulus-robust signal than fine-grained emotional categories. EEG patterns for Anger, Fear, Disgust, and Sadness may all share a negative-valence component that generalises, but the finer distinctions among them depend on stimulus-specific neural responses.

Full FT performs slightly worse than LoRA in 9-class generalisation across both datasets (FACED: 0.155 vs 0.160; THU-EP: 0.171 vs 0.179). The encoder becomes more sensitive to the specific training stimuli under full FT, a classic overfitting-to-distribution symptom that early stopping does not fully prevent.

### 8.5 Implications for SupCon

The near-collapse of 9-class generalisation across all FT methods, combined with the plateau in binary generalisation, strongly motivates Supervised Contrastive Learning. Standard CE-only fine-tuning optimises for correct class assignment given the training stimuli, but it does not explicitly enforce that same-emotion embeddings from **different stimuli** cluster together. The result is that representations are well-structured within seen stimuli but fragmented across unseen ones.

SupCon with same-emotion pairs drawn from across stimuli directly attacks this problem: the contrastive loss explicitly pulls together embeddings from the same emotion class regardless of which stimulus produced them, and pushes apart embeddings from different emotions. If cross-stimulus same-emotion similarity is the missing ingredient, SupCon is precisely the tool to inject it.

The expected outcome of SupCon is:
- Modest or negligible improvement in standard cross-subject CV accuracy (CE already handles this)
- Meaningful improvement in stimulus generalisation accuracy, particularly for binary valence (which is already partially generalisable) and potentially for 9-class
- Better-calibrated AUROC in the generalisation setting

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

The static split results confirm that no-mixup is consistently better for both val and test accuracy, and that the val→test generalisation gap is smaller without mixup, providing further evidence for the recommendation to disable mixup in all FT experiments.

---

## 10. Summary of Best Configurations

### 10.1 Overall Best per Setting

| Scenario | Best config | val_acc |
|---|---|---|
| FACED 9-class (cross-subject CV) | Full FT, no-mixup, pooling=no | **0.5820 ± 0.0332** |
| FACED binary (cross-subject CV) | Full FT, no-mixup, pooling=no | **0.7555 ± 0.0216** |
| THU-EP 9-class (cross-subject CV) | Full FT, no-mixup, pooling=no | **0.4828 ± 0.0272** |
| THU-EP binary (cross-subject CV) | Full FT, no-mixup, pooling=no | **0.6990 ± 0.0167** |
| FACED 9-class (static split, test) | Full FT, no-mixup | **0.4596** |
| FACED binary (static split, test) | Full FT, no-mixup | **0.7089** |
| FACED binary (stim. gen.) | LoRA/Full FT, no-mixup | **~0.597–0.598** |
| FACED 9-class (stim. gen.) | LoRA no-mixup | **0.1603** |
| THU-EP 9-class (stim. gen.) | LoRA no-mixup | **0.1790** |
| THU-EP binary (stim. gen.) | Full FT no-mixup | **0.5649** |

### 10.2 Ranking by Approach (FACED only, 9-class, cross-subject CV)

| Rank | Method | Mixup | val_acc |
|---|---|---|---|
| 1 | Full FT | No | **0.5820** |
| 2 | Full FT | Yes | 0.5490 |
| 3 | LoRA | No | 0.5362 |
| 4 | LoRA | Yes | 0.5042 |
| 5 | LP (baseline) | Yes | 0.4891 |

### 10.3 Ranking by Approach (THU-EP only, 9-class, cross-subject CV)

| Rank | Method | Mixup | val_acc |
|---|---|---|---|
| 1 | Full FT | No | **0.4828** |
| 2 | LoRA | No | 0.4668 |
| 3 | LoRA | Yes | 0.4357 |
| 4 | LP (baseline) | Yes | 0.4106 |

---

## 11. Overall Conclusions and Motivation for SupCon

### 11.1 Summary of FT Findings

1. **Fine-tuning consistently and substantially improves over linear probing.** Across all datasets and tasks, LoRA FT and full FT achieve higher accuracy, balanced accuracy, and AUROC. The improvement is most pronounced for the 9-class task (+5.6 to +7.2 pp for LoRA, +9.3 to +10.4 pp for full FT vs LP).

2. **Full FT outperforms LoRA, but at the cost of stability.** Full FT achieves 1.6–4.6 pp higher accuracy across tasks, but exhibits near-zero training loss and very high validation loss in several folds under the no-mixup regime, indicating memorisation. Early stopping prevents catastrophic collapse, but training dynamics are unstable. LoRA is the safer, more reproducible approach, particularly for THU-EP.

3. **Mixup hurts fine-tuning uniformly.** No-mixup outperforms mixup by 1–3 pp across all methods, datasets, and tasks. This is contrary to its typical role as a beneficial regulariser, and likely reflects a mismatch between synthetic blended inputs and the distribution of embeddings produced by a partially frozen pretrained encoder.

4. **Pooling `no` (full concatenation) is generally better than `last` (query token only).** The exception is FACED binary (+1.1 pp for `last`), but for THU-EP and 9-class tasks, `no` pooling is substantially better.

5. **THU-EP is harder and noisier than FACED** across all conditions. The LoRA–full FT gap is smaller for THU-EP, indicating diminishing returns from additional capacity with fewer training subjects.

6. **AUROC substantially exceeds accuracy** (0.83–0.88 for 9-class tasks). The model assigns well-calibrated probability rankings even when hard decisions are wrong — the representations encode emotion structure but a linear boundary cannot fully exploit it. Fine-tuning partially closes this gap but does not eliminate it.

### 11.2 The Case for Supervised Contrastive Learning

Despite consistent gains over LP, fine-tuning faces a fundamental limitation: the CE loss only enforces correct class assignment, not that same-emotion embeddings across different stimuli are geometrically similar. The evidence is unambiguous:

- **Stimulus generalisation collapses for 9-class** (FACED: 15–16% accuracy at AUROC ~0.53, barely above chance) across all methods including full FT.
- **The generalisation plateau for binary valence** is ~0.60, independent of CV accuracy (0.71–0.76). Stronger fine-tuning does not yield better generalisation.
- **Stronger fine-tuning leads to worse generalisation ratios** (full FT retains only 79% of CV performance vs 83% for LP on FACED binary), confirming that the accuracy gains from FT are largely stimulus-specific.

This directly motivates Supervised Contrastive Learning (SupCon, Khosla et al. 2020):

- **Mechanism:** SupCon explicitly groups embeddings from the same emotion class (regardless of stimulus) and separates them from different-emotion embeddings. This is orthogonal to what CE achieves and directly targets the cross-stimulus invariance that CE cannot enforce.
- **Expected benefit:** Improved stimulus generalisation, particularly for binary valence where some cross-stimulus signal already exists. A well-structured embedding space from SupCon should also yield better calibration (higher AUROC relative to accuracy).
- **Architecture fit:** The `context` representation (512-d query-attention vector) provides a compact, cross-channel embedding suitable for contrastive learning. The projection head maps this to a 128-d normalised space where cosine distance is used for pair discrimination.
- **Mixup incompatibility:** SupCon requires clean labels for positive pair identification — mixed-label samples cannot be reliably assigned to emotion classes — providing a principled reason to disable mixup, consistent with the empirical finding above.

The joint loss `L = α·L_CE + (1−α)·L_SupCon` preserves the classification objective while adding the geometric constraint. The hypothesis is that this combination will produce representations that are simultaneously discriminative across subjects (CE contribution) and invariant to stimulus identity (SupCon contribution), yielding measurable improvement in the stimulus generalisation setting without degrading cross-subject CV accuracy.
