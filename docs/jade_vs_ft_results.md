# JADE vs FT Results — FACED

JADE = LoRA + SupCon (α·CE + (1-α)·SupCon), τ=0.1, repr=context, no mixup.
FT baseline = FT-LoRA no-mixup (fair comparison since JADE disables mixup).

---

## 1. Cross-subject CV (10-fold)

### Binary

| Method | α | Acc | Bal Acc | AUROC | F1 |
|---|---|---|---|---|---|
| FT-LoRA (no-mixup) | — | 72.50 | 72.50 | 78.43 | 72.37 |
| JADE-LoRA | 0.2 | 71.58 | 71.58 | 77.35 | 71.52 |
| JADE-LoRA | 0.5 | 71.90 | 71.90 | 77.50 | 71.79 |
| JADE-LoRA | 0.8 | 72.07 | 72.07 | 77.87 | 72.02 |
| FT-FullFT (no-mixup) | — | 75.55 | 75.55 | 82.30 | 75.45 |
| JADE-FullFT | 0.1 | 76.82 | 76.82 | 83.43 | 76.80 |
| JADE-FullFT | 0.2 | **76.92** | **76.92** | **83.66** | **76.87** |
| JADE-FullFT | 0.3 | 76.47 | 76.47 | 83.15 | 76.43 |
| JADE-FullFT | 0.5 | 76.59 | 76.59 | 82.74 | 76.57 |
| JADE-FullFT | 0.7 | 75.93 | 75.93 | 82.66 | 75.89 |
| JADE-FullFT | 0.8 | 75.16 | 75.16 | 82.00 | 75.08 |

### 9-class

| Method | α | Acc | Bal Acc | AUROC | F1 |
|---|---|---|---|---|---|
| FT-LoRA (no-mixup) | — | 53.62 | 53.73 | 85.61 | 53.61 |
| JADE-LoRA | 0.5 | 51.45 | 51.52 | 84.34 | 51.31 |
| FT-FullFT (no-mixup) | — | 58.20 | 58.44 | 88.36 | 58.20 |
| JADE-FullFT | 0.1 | 57.20 | 57.44 | 88.05 | 57.17 |
| JADE-FullFT | 0.2 | 57.21 | 57.39 | 87.96 | 57.22 |
| JADE-FullFT | 0.3 | **58.81** | **59.03** | **88.61** | **58.90** |
| JADE-FullFT | 0.5 | 57.29 | 57.47 | 87.86 | 57.27 |
| JADE-FullFT | 0.7 | 57.27 | 57.51 | 88.06 | 57.21 |
| JADE-FullFT | 0.8 | 57.14 | 57.42 | 88.05 | 57.15 |
| JADE-FullFT | 0.9 | 57.61 | 57.87 | 88.35 | 57.62 |

---

## 2. Stimulus Generalization (3-seed cross-seed mean)

### Binary

| Method | α | Acc | Bal Acc | F1 |
|---|---|---|---|---|
| FT-LoRA (no-mixup) | — | 59.70 | 59.70 | 59.16 |
| JADE-LoRA | 0.2 | **59.85** | **59.85** | 59.17 |
| JADE-LoRA | 0.5 | 59.84 | 59.84 | **59.49** |
| JADE-LoRA | 0.8 | 59.61 | 59.61 | 59.10 |
| FT-FullFT (no-mixup) | — | 59.73 | 59.73 | 58.82 |
| JADE-FullFT | 0.5 | — | — | — |

### 9-class

| Method | α | Acc | Bal Acc | F1 |
|---|---|---|---|---|
| FT-LoRA (no-mixup) | — | 16.03 | 16.03 | **14.02** |
| JADE-LoRA | 0.5 | **16.37** | **16.37** | 13.83 |
| FT-FullFT (no-mixup) | — | 15.52 | 15.52 | 13.23 |
| JADE-FullFT | 0.5 | — | — | — |

---

*Notes: JADE-FullFT generalization runs not yet submitted. 9-class generalization near chance (1/9 ≈ 11.11%) for all methods. Binary fullft alpha=0.9 not yet run.*
