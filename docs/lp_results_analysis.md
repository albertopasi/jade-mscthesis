# Linear Probing — Ablations and Analysis

LP (frozen REVE encoder + trainable `cls_query_token` + linear head, `pooling=no`, w=10 s, no-mixup) is the lower-bound baseline used throughout the thesis. The headline LP accuracies (71.64 % binary, 50.27 % 9-class on FACED) are reported in the README and in [`results_brief.md`](results_brief.md).

This doc covers what those numbers *don't* show: the **LP-specific design ablations** (pooling mode, window size, mixup) that justify the configuration choices, and a representational observation (the AUROC–accuracy dissociation) that motivated the move to fine-tuning.

For the LP CV / generalization numbers themselves see [`results_brief.md`](results_brief.md) and [`statistical_tests.md`](statistical_tests.md).

---

## 1. The AUROC–accuracy dissociation

A consistent pattern in LP results: **AUROC is much higher than accuracy** on the 9-class task.

| Setting | Accuracy | Macro AUROC | Gap |
|---|---:|---:|---:|
| FACED 9-class, CV | 50.27 % | 83.35 % | 33.08 pp |
| FACED 9-class, stimulus-generalization | 15.83 % | ~54 % | ~38 pp |

What this means:

- The encoder **ranks the correct class highly** even when `argmax` is wrong. In a 9-class task, being the 2nd-most-probable class counts as wrong for accuracy but is captured by AUROC.
- **Adjacent emotions are genuinely confusable** in EEG (Anger vs Fear, Joy vs Amusement). The model lands in the right emotion neighbourhood but struggles at fine boundaries.
- The **linear head is a bottleneck** — the representations already contain the necessary information (high AUROC), but a linear boundary cannot exploit it fully. Encoder fine-tuning should close part of this gap.

This observation is what motivates the LP → SFT step in the thesis. SFT does close the gap somewhat (88.52 % AUROC, 58.52 % accuracy on 9-class); JADE closes it further (90.18 % AUROC, 62.03 % accuracy).

---

## 2. Ablation — Pooling mode (`no` vs `last`)

`pool_no` (the default) concatenates the query token with all channel × patch embeddings and flattens to a high-dimensional vector before the linear head. `pool_last` applies query attention and squeezes to a single 512-d vector.

**Head input dimensionality on FACED (32 channels, 11 patches per 10 s window):**

| Pooling | Head input dim | Reduction vs `no` |
|---|---:|---:|
| `no` (default) | (1 + 32 × 11) × 512 = 180,736 | — |
| `last` | 512 | 354× compression |

### Results (FACED, 10-fold CV)

| Task | `pool_no` (baseline) | `pool_last` | Δ Acc | Δ AUROC |
|---|---:|---:|---:|---:|
| Binary | 70.57 % ± 1.56 % | 61.49 % ± 2.47 % | **−9.08 pp** | −12.64 pp |
| 9-class | 48.91 % ± 2.86 % | 26.91 % ± 2.65 % | **−22.00 pp** | −15.96 pp |

(LP numbers in this table use the older `--with-mixup` configuration; the thesis-final no-mixup LP at `pool_no` is 71.64 / 50.27. The relative ordering is unchanged.)

`pool_last` was run with `max_epochs=50, patience=10` — a strictly longer training horizon than the `pool_no` baseline — and still trails by a large margin.

### Why the gap is so large

`pool_no` gives the linear head direct access to every channel's patch-level representation: 32 × 11 = 352 spatial-temporal positions, each contributing its own 512-d feature, so the classifier can learn channel-specific and time-specific weights. This matters for EEG emotion recognition where discriminative information is unevenly distributed across brain regions and time (frontal asymmetry for valence, temporal dynamics for arousal).

`pool_last` forces all of that through the query-attention bottleneck. Although the `cls_query_token` is trainable and can learn to attend to emotion-relevant positions, a single 512-d output cannot represent the full spatial-temporal pattern. For a linear head — no capacity to "unmix" compressed features — this is a hard ceiling.

**9-class suffers disproportionately** (−22 pp vs −9 pp on binary): valence is a coarse signal that survives aggressive compression; distinguishing nine emotions requires the detail `pool_no` preserves.

**AUROC drops too** (−13 to −16 pp): the 512-d query-attention vector loses *representational quality*, not just decision-boundary precision.

**Conclusion**: `pool_no` is the right default for LP. The thesis uses it for LP, SFT, and JADE.

---

## 3. Ablation — Window size (6 s vs 10 s)

FACED was re-run with a 6-second non-overlapping window (`--window 6 --stride 6`, 1200 pts).

| Task | w = 10 s (baseline) | w = 6 s | Δ Acc | Δ AUROC |
|---|---:|---:|---:|---:|
| Binary | 70.57 % ± 1.56 % | 64.68 % ± 1.47 % | **−5.89 pp** | −7.37 pp |
| 9-class | 48.91 % ± 2.86 % | 35.83 % ± 2.27 % | **−13.08 pp** | −8.97 pp |

REVE patch counts:
- w = 10 s (2000 pts): `floor((2000 − 200) / 180) + 1 = 11` patches per channel
- w = 6 s (1200 pts): `floor((1200 − 200) / 180) + 1 = 6` patches per channel

Despite producing more windows per stimulus (5 vs 3 in a 30 s trial), 6 s is substantially worse — fine-grained emotion classification needs the temporal context that only emerges over longer windows. **9-class is far more sensitive** to context length than binary (−13 pp vs −6 pp).

**Conclusion**: w = 10 s is the right choice. The thesis uses it everywhere.

---

## 4. Ablation — Mixup on vs off

Mixup was disabled (`--no-mixup`) and the same configuration re-run.

| Task | With Mixup | No Mixup | Δ Acc | Δ AUROC |
|---|---:|---:|---:|---:|
| Binary | 70.57 % ± 1.56 % | **71.50 % ± 1.58 %** | +0.93 pp | +0.86 pp |
| 9-class | 48.91 % ± 2.86 % | **49.72 % ± 2.93 %** | +0.81 pp | −0.48 pp |

All differences are within one standard deviation of fold variance — mixup has no meaningful effect in this LP setting.

Why: only `cls_query_token` + linear head are trainable (~1.7 M parameters), with the encoder fully frozen. Overfitting risk is low and the model cannot memorise training samples in the way mixup was designed to mitigate. The frozen encoder already acts as a strong implicit regulariser.

**Conclusion**: mixup is disabled in LP. It is also disabled in JADE for an unrelated reason (incompatibility with SupCon's positive-pair identification — see [`jade_approach_design.md`](jade_approach_design.md) §7), so for parity SFT also runs no-mixup. This makes "no-mixup" the consistent norm across all three approaches in the thesis.

---

## 5. Summary

- `pool_no` is the right pooling mode. `pool_last` loses ~22 pp on 9-class.
- w = 10 s is the right window. w = 6 s loses ~13 pp on 9-class.
- Mixup is neutral for LP; disabled for parity with SFT and JADE.
- The **AUROC–accuracy gap** under LP shows the encoder already encodes useful emotion structure — the linear head is the bottleneck. Fine-tuning the encoder closes part of the gap (SFT, JADE).
