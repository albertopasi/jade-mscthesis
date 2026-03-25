# Fine-Tuning with LoRA — Implementation Notes

## Overview

Two-stage downstream pipeline: **LP warmup** (frozen encoder) followed by **LoRA fine-tuning** (adapter-augmented encoder). This document compares our implementation against the official REVE codebase.

---

## Architecture

### Pipeline flow

```
Stage 1: LP warmup                    Stage 2: LoRA fine-tuning
┌─────────────────────────┐           ┌──────────────────────────────┐
│ REVE encoder (FROZEN)   │           │ REVE encoder + LoRA adapters │
│         ↓               │           │         ↓                    │
│ Query attention pooling  │   ──→    │ Query attention pooling       │
│         ↓               │           │         ↓                    │
│ RMSNorm → Dropout → FC  │           │ RMSNorm → Dropout → FC       │
│   (trainable head)      │           │   (trainable head)           │
└─────────────────────────┘           └──────────────────────────────┘

Trainable: cls_query_token + head     Trainable: LoRA adapters + cls_query_token + head
```

### LoRA targets

In each of the 22 transformer layers, LoRA adapters wrap the attention projections:

```
transformer.layers.{0..21}.0.to_qkv   Linear(512 → 1536, bias=False)
transformer.layers.{0..21}.0.to_out   Linear(512 → 512,  bias=False)
```

Total: 44 LoRA adapter pairs (A + B matrices per target).

---

## File mapping: official → ours

| Official file | Our file | What it does |
|---|---|---|
| `reve_official/src/dt.py` | `src/approaches/fine_tuning/train_ft.py` | Main entry point, two-stage loop |
| `reve_official/src/models/lora.py` | `src/approaches/fine_tuning/lora.py` | LoRA config + `get_peft_model` wrapping |
| `reve_official/src/models/classifier.py` | `src/approaches/fine_tuning/model.py` | Classifier (encoder + pooling + head) |
| `reve_official/src/utils/model_utils.py` | `model.py` (`freeze_encoder` / `unfreeze_encoder`) | Freeze/unfreeze helpers |
| `reve_official/src/configs/task/faced.yaml` | `src/approaches/fine_tuning/config.py` | Hyperparameters |
| `reve_official/src/configs/config_dt.yaml` | `config.py` (LoRA + trainer sections) | Global config (LoRA rank, clip_grad, etc.) |
| `reve_official/src/utils/optim.py` | Reused from `src/approaches/linear_probing/stable_adamw.py` | StableAdamW optimizer |

---

## Hyperparameter comparison

Values sourced from `reve_official/src/configs/task/faced.yaml` and `reve_official/src/configs/config_dt.yaml`.

| Parameter | Official LP | Official FT | Ours LP | Ours FT |
|---|---|---|---|---|
| Max epochs | 20 | 200 | 20 | 200 |
| Learning rate | 5e-3 | 1e-4 | 5e-3 | 1e-4 |
| Dropout | 0.05 | 0.1 | 0.05 | 0.1 |
| Warmup epochs | 3 | 5 | 3 | 5 |
| Scheduler patience | 6 | 6 | 5 (LP) | 6 (FT) |
| Early stop patience | 10 | 10 | 10 | 10 |
| Grad clip | 2.0 (global) | 2.0 (global) | 100.0 | 2.0 |
| Mixup | Yes | Yes | Yes | Yes |
| Batch size | 64 | 64 | 64 | 64 |
| Optimizer | StableAdamW | StableAdamW | StableAdamW | StableAdamW |
| Optimizer betas | (0.92, 0.999) | (0.92, 0.999) | (0.92, 0.999) | (0.92, 0.999) |
| Weight decay | 0.01 | 0.01 | 0.01 | 0.01 |
| LoRA rank | 16 | 16 | 16 | 16 |
| LoRA alpha | 16 (= rank) | 16 (= rank) | 16 (= rank) | 16 (= rank) |
| LoRA dropout | 0.0 | 0.0 | 0.0 | 0.0 |
| LoRA targets | attention | attention | attention | attention |
| AMP dtype | fp16 | fp16 | fp16 | fp16 |

### Intentional differences

1. **LP grad clip = 100.0 vs official 2.0**: The official `config_dt.yaml` sets `trainer.clip_grad: 2.0` globally for both stages. However, the FACED task config also has `clip: 100` (unused in dt.py). Our standalone LP approach uses 100.0 and has been validated, so we keep 100.0 for the LP warmup stage and use 2.0 for the FT stage only.

2. **LP scheduler patience = 5 vs official 6**: Our LP implementation uses patience=5 (matching our standalone LP config). The FT stage uses 6 (matching official FT config). This is consistent with our existing LP results.

---

## Key implementation differences

### 1. LoRA wrapping scope

**Official** (`reve_official/src/models/lora.py:76`): wraps the **entire** `ReveClassifier` model with peft.
```python
# Official: wraps the whole classifier
model = get_peft_model(model, config)  # model = ReveClassifier
```
This means peft freezes ALL base params (including `linear_head` and `cls_query_token`). With `train_all=False` (the default), only LoRA adapter weights are trainable during FT. The head and query token are frozen.

**Ours** (`src/approaches/fine_tuning/lora.py:71`): wraps only `model.reve` (the encoder).
```python
# Ours: wraps only the encoder
model.reve = get_peft_model(model.reve, config)
```
This leaves `cls_query_token` and `linear_head` untouched by peft — they remain trainable from the LP stage. During FT, trainable params = LoRA adapters + cls_query_token + linear_head.

**Why this differs**: In the official code, it appears that only LoRA params are trained during FT (head and cls_query are frozen by peft). This may be intentional or a subtle bug — the `CustomGetLora.train_all=False` flag suggests they intended this. Our approach trains the head and query token through both stages, which is more intuitive and matches the plan's intent.

### 2. Model class separation

**Official**: Single `ReveClassifier` class used for both LP and FT. Forward pass has no `torch.no_grad()` — frozen/unfrozen state is controlled purely by `requires_grad` flags.

**Ours**: Separate `ReveClassifierLP` (uses `torch.no_grad()` in forward) and `ReveClassifierFT` (no `torch.no_grad()`, controlled by `requires_grad`). This keeps each class focused on its purpose. The LP class can never accidentally allow gradients through the encoder, while the FT class is designed for gradient flow.

### 3. Training loop structure

**Official** (`reve_official/src/dt.py`): Uses Hydra config system, DDP for multi-GPU, `idr_torch` for distributed coordination. The `train_stage()` function is called once per stage with different configs.

**Ours**: No Hydra, no DDP. Single `train_stage()` function parameterized by keyword arguments (lr, epochs, patience, etc.), called twice within `run_fold_ft()`. CLI arguments via `argparse`.

### 4. Best model restoration between stages

**Official**: Saves `model_best.pth` to disk during training, then the model state at end of LP carries into FT. There is no explicit "restore best LP weights before FT" step — the model simply continues from wherever it ended up.

**Ours**: After LP stage, we restore the best LP state before entering FT (`model.load_state_dict(lp_result["best_state"])`). This ensures the FT stage starts from the best LP checkpoint, not the last epoch (which may have degraded due to overfit).

### 5. LoRA target module names (bug fix)

**Official** (`reve_official/src/models/lora.py:37-38`): Inconsistent module path prefixes:
```python
# Official — note the missing "transformer." prefix on to_out
target_modules.extend([
    f"transformer.layers.{i}.0.to_qkv",  # ← has "transformer."
    f"layers.{i}.0.to_out"               # ← missing "transformer."
])
```
This works because peft matches by suffix, so `layers.{i}.0.to_out` still matches `transformer.layers.{i}.0.to_out`. But it's inconsistent.

**Ours**: Uses correct full paths for both:
```python
target_modules.append(f"transformer.layers.{i}.0.to_qkv")
target_modules.append(f"transformer.layers.{i}.0.to_out")
```

### 6. Evaluation metric for early stopping / best model

**Official** (`reve_official/src/dt.py:133`): Tracks best model by **balanced accuracy** (`val_balanced_acc`).

**Ours**: Tracks best model by **accuracy** (`val_acc`), consistent with our LP implementation. Both metrics are logged; the choice of tracking metric can be changed if needed.

### 7. Checkpoint saving

**Official**: Saves the entire `model.state_dict()` as `model_best.pth`.

**Ours**: Saves LoRA adapter weights separately via `model.reve.save_pretrained()` (peft convention) plus head + cls_query_token via `torch.save()`. This is more storage-efficient — only the adapted/trainable parts are saved, not the full 85M-param encoder.

### 8. Warmup schedule boundary

**Official** (`reve_official/src/dt.py:109`): `warmup = epoch < warmup_epochs` (exclusive — warmup active for epochs 0..N-1).

**Ours**: `warmup_active = epoch <= warmup_epochs` (inclusive — warmup active for epochs 0..N). This matches our existing LP implementation and means 1 extra epoch of warmup. Consistent across our codebase.

---

## What's reused from LP (imported, not duplicated)

| Component | Import path |
|---|---|
| `load_reve_and_positions()` | `src.approaches.linear_probing.model` |
| `RMSNorm` | `src.approaches.linear_probing.model` |
| `compute_n_patches()` | `src.approaches.linear_probing.model` |
| `evaluate_model()` | `src.approaches.linear_probing.model` |
| `StableAdamW` | `src.approaches.linear_probing.stable_adamw` |
| `build_raw_dataset()` | `src.approaches.linear_probing.train_lp` |
| `get_channel_names()` | `src.approaches.linear_probing.train_lp` |
| `PatienceMonitor` | `src.approaches.linear_probing.train_lp` |
| `_get_exponential_warmup_lambda()` | `src.approaches.linear_probing.train_lp` |
| `fmt_dur()`, `fmt_metric()`, `COL_W` | `src.approaches.linear_probing.train_lp` |
| Fold splits | `src.datasets.folds` |
| Dataset classes | `src.datasets.faced_dataset`, `src.datasets.thu_ep_dataset` |

`build_raw_dataset()` is duck-typed — it accesses `.dataset`, `.task_mode`, `.data_root`, `.window_size`, `.stride`, `.scale_factor`, all of which exist on both `LPConfig` and `FTConfig`.

---

## SupCon extensibility

The architecture is designed for a future SupCon loss addition:

```
                    ┌─── projection_head → SupConLoss
                    │
encoder → pooling ──┤
                    │
                    └─── linear_head → CrossEntropyLoss

total_loss = w_ce * CE + w_sc * SupCon
```

The `forward()` method can be extended to return both logits and projected embeddings. The `train_stage()` loss computation is isolated, making it easy to augment with a second loss term.

---

## CLI reference

```bash
# Full run, all 10 folds
uv run python -m src.approaches.fine_tuning.train_ft \
    --dataset faced --task 9-class --pooling no

# Single fold, custom LoRA rank
uv run python -m src.approaches.fine_tuning.train_ft \
    --dataset thu-ep --task binary --fold 3 --lora-rank 8

# Smoke test (minimal epochs)
uv run python -m src.approaches.fine_tuning.train_ft \
    --dataset faced --task 9-class --fold 1 --lp-epochs 2 --ft-epochs 3

# Override FT learning rate
uv run python -m src.approaches.fine_tuning.train_ft \
    --dataset faced --ft-lr 5e-5 --ft-epochs 100

# Generalization mode
uv run python -m src.approaches.fine_tuning.train_ft \
    --dataset faced --task binary --generalization --gen-seeds 123 456 789

# Attention + FFN LoRA targets
uv run python -m src.approaches.fine_tuning.train_ft \
    --dataset faced --lora-target attention+ffn
```

---

## Output structure

```
outputs/ft_checkpoints/
└── ft_faced_9-class_w10s10_pool_no_r16_fold_1/
    ├── lora_adapter/           # peft adapter weights (adapter_config.json + adapter_model.safetensors)
    ├── head_weights.pt         # cls_query_token + linear_head state dict
    └── ...
```
