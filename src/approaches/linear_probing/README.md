# Linear Probing

REVE linear probing (LP) downstream task.

Frozen REVE encoder + trainable `cls_query_token` + trainable linear head.

---

## How it works

1. Raw EEG windows `(B, C, T)` are fed to the frozen REVE encoder, which returns patch embeddings `(B, C, H, E)`.
2. A trainable query token attends over the patches to produce a context vector.
3. Depending on the pooling mode, the context (and optionally all patches) are flattened and passed to a small linear head.
4. Only the query token and linear head are updated during training (~180K params for pooling="no", FACED binary).

---

## Files

| File | Description |
|---|---|
| `config.py` | `LPConfig` dataclass — all paths, hyperparameters, derived helpers |
| `model.py` | `ReveClassifierLP` (official mode), `EmbeddingExtractor`, `LinearProber` (fast mode) |
| `train_lp.py` | Main entry point — CLI, training loops, fold orchestration |
| `stable_adamw.py` | Port of official `StableAdamW` optimizer (debiased betas, RMS-stabilized LR) |
| `summary.py` | Cross-fold summary printing and JSON export |

---

## Modes

### Official mode (default)
Frozen encoder runs live each batch. Trainable: `cls_query_token` + linear head.
Matches the official REVE LP procedure.

### Fast mode (`--fast`)
Pre-computes REVE embeddings once per subject, caches them to disk, then trains only the linear head. Faster iteration but not a faithful reproduction (no trainable query token during training).

---

## Pooling modes

| Mode | Description | Output dim (FACED, 10s window) |
|---|---|---|
| `no` (default) | Query attention → concat [context, all patches] → flatten | (1 + 32×11) × 512 = 180,736 |
| `last` | Query attention → squeeze | 512 |
| `last_avg` | Mean pool across patches | 512 |

---

## Running

```bash
# FACED, 9-class, all 10 folds (official mode)
uv run python -m src.approaches.linear_probing.train_lp --dataset faced --task 9-class

# THU-EP, binary, single fold
uv run python -m src.approaches.linear_probing.train_lp --dataset thu-ep --task binary --fold 3

# Generalization mode (2/3 train stimuli, 1/3 held-out)
uv run python -m src.approaches.linear_probing.train_lp --dataset faced --generalization

# Disable mixup or AMP
uv run python -m src.approaches.linear_probing.train_lp --no-mixup --no-amp

# Fast mode
uv run python -m src.approaches.linear_probing.train_lp --dataset faced --fast
```

---

## Training details (official mode)

| Setting | Value |
|---|---|
| Optimizer | StableAdamW (β=[0.92, 0.999], wd=0.01, eps=1e-9) |
| LR | 5e-3 |
| LR schedule | Exponential warmup (3 epochs) + ReduceLROnPlateau (patience=5, factor=0.5) |
| Max epochs | 50 (early stop patience=15) |
| Batch size | 64 |
| Augmentation | Mixup λ~U(0,1) |
| Precision | AMP float16 |
| Grad clip | max_norm=100 |
| Dropout | 0.05 |
| Scale factor | 1000 (µV → mV-range) |

---

## Outputs

- Checkpoints: `outputs/lp_checkpoints/<run_name>/classifier_weights.pt`
- W&B project: Set `USE_WANDB = False` in `config.py` to disable.
- Metrics: accuracy, balanced accuracy, AUROC (macro OvR for 9-class), weighted F1
