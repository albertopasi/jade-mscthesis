# JADE vs FT Results — FACED

JADE = SupCon joint loss `α·CE + (1-α)·SupCon`, repr=context, no mixup.
FT baseline = FT-FullFT no-mixup (fair comparison — JADE disables mixup).
All rows: 10-fold cross-subject CV, val metrics × 100.
**Δ Acc** = JADE acc − FT-FullFT acc.

---

## Binary

FT-FullFT no-mixup baseline: **Acc=75.55 · Bal=75.55 · AUROC=82.30 · F1=75.45**

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
| 0.3 | 0.5 | — | — | — | — | (pending) |
| 0.5 | 0.1 | 76.59 | 76.59 | 82.74 | 76.57 | +1.04 |
| 0.7 | 0.1 | 75.93 | 75.93 | 82.66 | 75.89 | +0.38 |
| 0.8 | 0.1 | 75.16 | 75.16 | 82.00 | 75.08 | −0.39 |

**Every JADE config except α=0.8 beats the FT baseline.** Best: α=0.2, τ=0.05 (+1.73pp).

---

## 9-class

FT-FullFT no-mixup baseline: **Acc=58.20 · Bal=58.44 · AUROC=88.36 · F1=58.20**

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

**Only α=0.3, τ=0.1 beats FT baseline** (+0.61pp). All other configs marginally below. 9-class is much less robust to SupCon HPs than binary.

---

## Summary — binary vs 9-class behaviour

| Task | FT-FullFT baseline | JADE best | Δ | Winning config |
|---|:---:|:---:|:---:|---|
| Binary | 75.55 | **77.28** | +1.73 | α=0.2, τ=0.05 |
| 9-class | 58.20 | **58.81** | +0.61 | α=0.3, τ=0.1 |

- **Binary:** SupCon robustly helps across the full (α, τ) grid. Broad optimum near α=0.2.
- **9-class:** SupCon barely helps, narrow optimum; easy to hurt with wrong τ. Harder task, smaller positive sets per anchor → weaker contrastive signal.

---

## Pending / still running

- τ=0.03 sweep (4 jobs) — submitted, not yet reported
- Binary α=0.3, τ=0.5 — still running / pending
- JADE-FullFT generalization runs — not yet submitted

## LoRA reference (earlier sweep, not the focus)

LoRA-JADE never beat FT-LoRA in cross-subject CV. Numbers retained in earlier revisions of this doc if needed.
