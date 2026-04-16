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
| JADE-FullFT | 0.5 | **76.59** | **76.59** | **82.74** | **76.57** |

### 9-class

| Method | α | Acc | Bal Acc | AUROC | F1 |
|---|---|---|---|---|---|
| FT-LoRA (no-mixup) | — | 53.62 | 53.73 | 85.61 | 53.61 |
| JADE-LoRA | 0.5 | 51.45 | 51.52 | 84.34 | 51.31 |
| FT-FullFT (no-mixup) | — | **58.20** | **58.44** | **88.36** | **58.20** |
| JADE-FullFT | 0.5 | 57.29 | 57.47 | 87.86 | 57.27 |

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

*Notes: JADE-FullFT generalization runs not yet submitted. 9-class generalization near chance (1/9 ≈ 11.11%) for all methods.*
