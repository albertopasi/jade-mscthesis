# Fine-Tuning

Two-stage fine-tuning of the REVE encoder on EEG emotion classification.

**Stage 1 (LP warmup):** frozen encoder, train `cls_query_token` + linear head — identical to linear probing.
**Stage 2 (FT):** unfreeze encoder and apply LoRA adapters (default) or full fine-tuning (`--fullft`).

---

## Files

| File | Description |
|---|---|
| `config.py` | `FTConfig` dataclass — all paths, hyperparameters, LoRA settings, derived helpers |
| `model.py` | `ReveClassifierFT` — same architecture as LP but supports freeze/unfreeze of encoder |
| `lora.py` | `apply_lora`, `print_lora_summary` — peft LoRA injection into REVE attention layers |
| `training.py` | `train_stage()` — generic training loop used by both LP warmup and FT stage |
| `summary.py` | Thin wrapper around `shared/summary.py` for FT-specific filenames |
| `train_ft.py` | Main entry point — CLI, fold orchestration, checkpoint saving |

---

## Evaluation modes (mutually exclusive)

| Flag | Mode | Subjects split | Output |
|---|---|---|---|
| *(default)* | 10-fold cross-subject CV | KFold(n=10, seed=42) | `summary_*.json` |
| `--revesplit` | Official REVE static split (FACED only) | train 0–79 / val 80–99 / test 100–122 | `summary_*_revesplit.json` |
| `--generalization` | Stimulus generalization | same k-fold subjects, 2/3 train stimuli / 1/3 held-out | `summary_*_gen_s{seed}.json` |

---

## Stage 2 modes

| Flag | Behaviour |
|---|---|
| *(default)* | LoRA adapters injected into `to_qkv` + `to_out` of all 22 transformer layers |
| `--fullft` | Entire encoder unfrozen, no LoRA — all encoder parameters updated |

---

## Running

```bash
# All folds, FACED, 9-class, LoRA (default)
uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --task 9-class

# THU-EP, binary, single fold
uv run python -m src.approaches.fine_tuning.train_ft --dataset thu-ep --task binary --fold 1

# Custom LoRA rank
uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --lora-rank 8

# Full fine-tuning (no LoRA)
uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --fullft

# Official REVE static split (includes held-out test set)
uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --revesplit
uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --fullft --revesplit

# Stimulus generalization (3 seeds by default)
uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --generalization

# Disable mixup or AMP
uv run python -m src.approaches.fine_tuning.train_ft --no-mixup --no-amp

# Smoke test
uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --fold 1 --lp-epochs 2 --ft-epochs 3
```

---

## Training details

| Setting | LP warmup stage | FT stage |
|---|---|---|
| Optimizer | StableAdamW (β=[0.92, 0.999], wd=0.01) | same |
| LR | 5e-3 | 1e-4 |
| LR schedule | Exp warmup (3 ep) + ReduceLROnPlateau (patience=6) | Exp warmup (5 ep) + ReduceLROnPlateau (patience=6) |
| Max epochs | 20 | 200 |
| Early stopping | patience=10, val_acc (`>`) | patience=20, val_acc (`>`) |
| Grad clip | max_norm=2.0 | max_norm=2.0 |
| Dropout | 0.05 | 0.1 |
| Batch size | 64 | 64 |
| Precision | AMP float16 | AMP float16 |
| Augmentation | Mixup λ~U(0,1) | Mixup λ~U(0,1) |
| Best model metric | val_acc | val_acc |

---

## LoRA configuration

- **Targets**: `transformer.layers.{i}.0.to_qkv` and `transformer.layers.{i}.0.to_out` (22 layers × 2 = 44 modules)
- **Default**: rank=16, alpha=16, dropout=0.0 (scaling factor alpha/rank = 1.0)
- **Library**: peft (`LoraConfig` + `get_peft_model`)

---

## Saved checkpoints

Checkpoints are saved under `outputs/ft_checkpoints/<run_name>/`.

- **LoRA mode**: `lora_adapter/` (peft convention) + `head_weights.pt` (cls_query_token + linear_head)
- **Full FT mode**: `full_model.pt` (complete state dict)
- **LP warmup best state** is held in memory and loaded with `strict=False` before stage 2 begins

---

## Outputs

- Checkpoints: `outputs/ft_checkpoints/<run_name>/`
- Summary JSON: `outputs/ft_checkpoints/summary_*.json`
- W&B project: `eeg-ft-v2` (set `USE_WANDB = False` in `config.py` to disable)
- Metrics: accuracy, balanced accuracy, AUROC (macro OvR for 9-class), weighted F1
