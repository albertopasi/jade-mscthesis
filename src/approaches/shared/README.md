# Shared Utilities

Common building blocks used by both the Linear Probing (LP) and Fine-Tuning (FT) pipelines. Neither approach should import utilities from the other — both import from here.

---

## Files

| File | Exports | Used by |
|---|---|---|
| `config.py` | `PROJECT_ROOT`, `DATA_ROOTS`, `REVE_MODEL_PATH`, `REVE_POS_PATH`, `SAMPLING_RATE`, `DEVICE`, `NUM_WORKERS`, `DATASET_DEFAULTS` | LP config, FT config |
| `reve.py` | `load_reve_and_positions`, `get_channel_names` | LP train, FT train |
| `model_utils.py` | `RMSNorm`, `compute_n_patches` | LP model, FT model |
| `metrics.py` | `evaluate_model` | LP model (re-export), FT training |
| `stable_adamw.py` | `StableAdamW` | LP train, FT training |
| `training_utils.py` | `fmt_dur`, `COL_W`, `_get_exponential_warmup_lambda` | LP train, FT training |
| `dataset.py` | `build_raw_dataset` | LP train, FT train |
| `summary.py` | `print_fold_summary`, `print_cross_seed_summary` | LP summary, FT summary |

---

## Key details

### `config.py`
Single source of truth for all paths and hardware constants. Approach-specific configs (`LPConfig`, `FTConfig`) import from here and add their own output dirs, W&B settings, and hyperparameter dataclasses.

### `reve.py`
Loads the pretrained REVE encoder and position bank from local paths. Called once per run in `main()`, and the resulting `(reve_model, pos_tensor)` is passed into each fold.

### `model_utils.py`
- `RMSNorm`: matches the RMSNorm used inside the REVE backbone.
- `compute_n_patches(window_size)`: `(window_size - 200) // 180 + 1` — e.g. 11 patches for a 2000-sample window.

### `metrics.py`
`evaluate_model` runs inference over a DataLoader and returns:
- `accuracy`: top-1
- `balanced_acc`: macro-averaged recall (sklearn)
- `f1_weighted`: weighted F1 (sklearn)
- `auroc`: OvR macro for 9-class, positive-class prob for binary
- `val_loss`: sample-weighted mean cross-entropy

### `stable_adamw.py`
Port of the official REVE StableAdamW: debiased betas + RMS-stabilized learning rates + decoupled weight decay.

### `training_utils.py`
- `fmt_dur`: formats seconds as `1h02m03s` / `5m10s` / `42s`
- `COL_W = 105`: column width for training log tables
- `_get_exponential_warmup_lambda`: exponential warmup matching the official REVE schedule — `(10^(step/total) - 1) / 9`

### `dataset.py`
`build_raw_dataset(cfg, subject_ids, stimulus_filter)` is duck-typed: any config with `.dataset`, `.task_mode`, `.data_root`, `.window_size`, `.stride`, `.scale_factor` works — both `LPConfig` and `FTConfig` qualify.

### `summary.py`
Prints a cross-fold metrics table and saves a JSON file. Called by the thin wrappers in `linear_probing/summary.py` and `fine_tuning/summary.py`, which supply approach-specific filenames and labels.
