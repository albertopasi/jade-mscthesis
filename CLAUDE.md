# CLAUDE.md — jade-mscthesis

MSc thesis project: reproduce and extend REVE for emotion recognition on EEG. Original plan was to compare LP, LoRA-FT, and JADE (LoRA + joint CE+SupCon). After extensive sweeps, the thesis has been refocused — see "Thesis status" section below.

## Thesis status (current)

**Thesis done.** The report PDF is at `MASTER_THESIS_REPORT(2).pdf`. The headline result is a new SOTA on FACED under standard 10-fold cross-subject CV: **JADE 76.32 % binary, 62.03 % 9-class** (vs prior best CL-CS at 72.5 / 43.4). Two findings:

1. **RQ1 — Frozen REVE + linear probe already matches or beats SOTA contrastive cross-subject frameworks** (LP > CL-CS by +6.87 pp on 9-class, parity on binary). Subject-invariance is inherited from large-scale pretraining; not learned from FACED.
2. **RQ2 — JADE (joint CE + SupCon) significantly improves over SFT on 9-class** (+3.51 pp, Wilcoxon Holm-adjusted p = 4.4e−10). Binary gain is +0.80 pp and *not* significant (p = 0.080) — the task is too coarse for label-aware SupCon to help.

**Final JADE configs on FACED (thesis Table 6.1):**
- 9-class: `α=0.3, τ=0.2, B=256, ft_lr=4e-4, fullft` → **62.03 ± 14.70**
- Binary: `α=0.2, τ=0.05, B=128, ft_lr=1e-4, fullft` → **76.32 ± 8.12**

These are the configs you'll see referenced everywhere; their JSONs live under `main-results/jade_{9-class,binary}/`.

**Scope decision**: FACED-only thesis. THU-EP was preprocessed and explored as a cross-dataset transfer sanity check but is not part of the main results, statistical tests, or generalization analysis. The README, statistical_tests outputs, and reproduction docs deliberately exclude THU-EP references.

**Stimulus-generalization (held-out subjects × held-out stimuli)**:
- Binary: ~59 % under all three methods (above 50 % chance), no significant ordering.
- 9-class: ~16 % under all three methods (near 11 % chance — task collapses).
- This is reported as a limitation, not solved.

**Generalization data status (current snapshot — check `main-results/*_generalization/` for ground truth):**
- JADE 9-class, JADE binary, FT 9-class, FT binary, LP 9-class: per-seed JSONs (123, 789) + `_gen_avg.json` written.
- LP binary: per-seed runs pending. The `statistical_tests_generalization.py` driver gracefully degrades to k=2 (Wilcoxon) for binary until LP binary lands.

**Important methodological norm (do not break)**: when a JADE config departs from the REVE recipe (e.g. B=256 with scaled LR), the SFT baseline **must** be re-run at the *same* recipe. Never compare JADE@(new recipe) to SFT@(old recipe). The thesis HP protocol is built around this; see `docs/jade_hp_sweep.md`.

**HP tuning insights worth remembering**:
- Single-fold val_acc has ~1 pp run-to-run variance. Single-fold rankings within ~1 pp are *not* interpretable as HP effects; full 10-fold CV is required for any conclusion.
- LR=8e-4 on 9-class gave the highest fold-1 val_acc but had cross-fold instability (training divergence on individual folds). Rejected for stability, not absolute accuracy.
- B=512 OOMs on A100-80GB; B=256 fits at ~62 GB peak.

**Run-name convention** (in `config.run_name_stem()` for all three approaches): summary JSON / checkpoint dir / W&B run name encode every distinguishing HP (`_b{batch}_lr{ft_lr:g}_{fullft|r{rank}}{_gen_s{seed}}`). Different recipes never collide on disk. If you add a new HP that distinguishes sweep cells, encode it in `run_name_stem` — otherwise two runs silently overwrite each other.

**Where to read next:**
- `README.md` — orientation + main results table + figures
- `docs/README.md` — annotated index over `docs/`
- `docs/architecture.md` — code design (training pipelines, inference contract, conventions)
- `docs/reproducing_results.md` — exact commands for every thesis result

## Always use `uv run python` to run anything (not plain `python`)

```bash
uv run python -m src.approaches.linear_probing.train_lp ...
```

When making changes to existing code, prefer minimal diffs that preserve the original logic. Do not refactor or over-engineer unless explicitly asked.

When adding results or data to documentation, default to summary/aggregate level (e.g., cross-seed averages) unless per-instance detail is explicitly requested.

Do not extensively explore the codebase or run smoke tests unless asked. Start implementing directly based on the information available, and ask clarifying questions if needed.

Before making any changes, create a todo list of exactly what you plan to do. Wait for my approval before starting. Keep each step small and specific.
---

## Project structure

```
jade-mscthesis/
├── configs/
│   └── thu_ep.yml                  # THU-EP dataset config (paths, channels, sampling)
├── data/
│   ├── FACED/
│   │   ├── Processed_data/         # Raw .pkl files from FACED download (28, 32, 7500)
│   │   └── preprocessed_v2/        # After preprocessing: sub{NNN}.npy (28, 32, 6000)
│   └── thu ep/
│       ├── EEG data/               # Raw .mat files (7500, 32, 28, 6)
│       └── preprocessed_v2/        # After preprocessing: sub_XX.npy (28, 30, 6000)
├── models/
│   └── reve_pretrained_original/
│       ├── reve-base/              # HuggingFace REVE model (modeling_reve.py, config.json, ...)
│       └── reve-positions/         # Position bank for EEG channels
├── outputs/
│   ├── lp_checkpoints/             # Saved LP classifier weights per fold/run
│   ├── ft_checkpoints/             # Saved FT LoRA weights + head + query token per fold/run
│   └── sc_checkpoints/             # Saved SC LoRA weights + head + query token + projection head per fold/run
├── reve_official/                  # Reference only — will be deleted after implementation
└── src/
    ├── approaches/
    │   ├── shared/
    │   │   ├── config.py           # PROJECT_ROOT, DATA_ROOTS, REVE paths, SAMPLING_RATE, DEVICE, NUM_WORKERS, DATASET_DEFAULTS
    │   │   ├── stable_adamw.py     # StableAdamW optimizer (used by LP + FT)
    │   │   ├── training_utils.py   # fmt_dur, COL_W, _get_exponential_warmup_lambda
    │   │   ├── metrics.py          # evaluate_model (accuracy, balanced_acc, AUROC, F1)
    │   │   ├── model_utils.py      # RMSNorm, compute_n_patches
    │   │   ├── dataset.py          # build_raw_dataset (duck-typed cfg, works for LP + FT)
    │   │   ├── reve.py             # load_reve_and_positions, get_channel_names
    │   │   └── summary.py          # print_fold_summary, print_cross_seed_summary (generic)
    │   ├── linear_probing/
    │   │   ├── config.py           # LPConfig dataclass + LP-specific paths/hyperparameters
    │   │   ├── model.py            # ReveClassifierLP, EmbeddingExtractor, LinearProber (evaluate_model → shared/metrics.py)
    │   │   ├── train_lp.py         # Main entry point — CLI + training loops
    │   │   ├── stable_adamw.py     # Legacy file (kept for reference, unused — shared/ is canonical)
    │   │   ├── summary.py          # Thin wrapper around shared summary for LP
    │   │   └── dataset.py          # EmbeddedDataset (fast mode only)
    │   ├── fine_tuning/
    │   │   ├── config.py           # FTConfig dataclass + LoRA/two-stage hyperparameters
    │   │   ├── model.py            # ReveClassifierFT
    │   │   ├── lora.py             # apply_lora, print_lora_summary (peft LoRA injection)
    │   │   ├── training.py         # train_stage() — shared loop for LP warmup + FT stages
    │   │   ├── summary.py          # Thin wrapper around shared summary for FT
    │   │   └── train_ft.py         # Main entry point — CLI + two-stage training loops
    │   └── supcon/
    │       ├── config.py           # SCConfig dataclass + SupCon hyperparameters
    │       ├── model.py            # ReveClassifierSC (FT model + projection head)
    │       ├── loss.py             # SupConLoss (Khosla et al. 2020, L_sup_out)
    │       ├── training.py         # train_stage_sc() — joint CE + SupCon training loop
    │       ├── summary.py          # Thin wrapper around shared summary for SC
    │       └── train_sc.py         # Main entry point — CLI + two-stage training loops
    ├── datasets/
    │   ├── faced_dataset.py        # FACEDWindowDataset
    │   ├── thu_ep_dataset.py       # THUEPWindowDataset (moved from src/thu_ep/)
    │   └── folds.py                # get_all_subjects, get_kfold_splits, get_stimulus_generalization_split
    ├── preprocessing/
    │   ├── faced/
    │   │   └── run_preprocessing.py  # pkl → npy, scipy.signal.resample 250→200 Hz
    │   └── thu_ep/
    │       ├── config.py             # THUEPConfig (reads configs/thu_ep.yml)
    │       ├── preprocessing_steps.py
    │       ├── thu_ep_preprocessing_pipeline.py
    │       └── run_preprocessing.py
    ├── exploration/                  # Data exploration scripts (not part of pipeline)
    └── utils/
        └── callbacks.py              # EpochSummaryCallback (Lightning)
```

---

## Datasets

### FACED
- **123 subjects** (0-indexed: 0–122), 32 channels, 28 stimuli, 9 emotions
- Raw: `.pkl` files `sub000.pkl`–`sub122.pkl`, shape `(28, 32, 7500)` at 250 Hz
- Preprocessed: `.npy` files, shape `(28, 32, 6000)` at 200 Hz (scipy FFT resample)
- Values stored in raw µV; divided by `scale_factor=1000` at load time

### THU-EP
- **80 subjects** (1-indexed: 1–80), subject 75 excluded (corrupted) → **79 usable**
  - Subject 37: exclude stimuli indices {15, 21, 24}
  - Subject 46: exclude stimuli indices {3, 9, 17, 23, 26}
- 30 channels (32 original minus A1, A2), 28 stimuli, 9 emotions
- Raw: `.mat` (HDF5) files, shape `(7500, 32, 28, 6)` at 250 Hz, 6 frequency bands
- Preprocessing: extract broadband (band index 5), remove A1/A2, scipy resample 250→200 Hz
- Preprocessed: `.npy` shape `(28, 30, 6000)` — raw µV values
- Values divided by `scale_factor=1000` at load time

### Shared label structure (both datasets)
28 stimuli → 9 emotion classes:
```
Stimuli  0-2:  Anger       → class 0  (neg in binary)
Stimuli  3-5:  Disgust     → class 1  (neg)
Stimuli  6-8:  Fear        → class 2  (neg)
Stimuli  9-11: Sadness     → class 3  (neg)
Stimuli 12-15: Neutral     → class 4  (dropped in binary)
Stimuli 16-18: Amusement   → class 5  (pos in binary)
Stimuli 19-21: Inspiration → class 6  (pos)
Stimuli 22-24: Joy         → class 7  (pos)
Stimuli 25-27: Tenderness  → class 8  (pos)
```

---

## REVE model

- Pretrained EEG foundation model at `models/reve_pretrained_original/reve-base`
- 22 transformer layers, `embed_dim=512`, 8 heads, `patch_size=200`, `overlap=20`
- Input: `(B, C, T)` EEG + position tensor
- `forward(eeg, pos, return_output=False)` → `(B, C, H, E)` where H = n_patches
- `cls_query_token`: trainable `nn.Parameter(torch.randn(1, 1, 512))` — **must remain trainable during LP**
- `n_patches = floor((window_size - patch_size) / (patch_size - overlap)) + 1`
  - For `window_size=2000`: n_patches = 11

---

## Linear Probing pipeline

### Step 1 — Preprocess

```bash
# FACED (123 subjects)
uv run python -m src.preprocessing.faced.run_preprocessing
uv run python -m src.preprocessing.faced.run_preprocessing --validate

# THU-EP (79 subjects)
uv run python -m src.preprocessing.thu_ep.run_preprocessing
uv run python -m src.preprocessing.thu_ep.run_preprocessing --validate
```

### Step 2 — Train

```bash
# Official mode (faithful reproduction) — frozen encoder + trainable cls_query_token
uv run python -m src.approaches.linear_probing.train_lp \
    --dataset faced \          # faced | thu-ep
    --task 9-class \           # 9-class | binary
    --pooling no \             # no | last | last_avg
    --window 2000 \            # timepoints (2000 = 10s at 200Hz)
    --stride 2000              # non-overlapping by default

# Fast mode (pre-computed embeddings, not official)
uv run python -m src.approaches.linear_probing.train_lp --dataset faced --fast

# Single fold
uv run python -m src.approaches.linear_probing.train_lp --dataset faced --fold 3

# Generalization evaluation (2/3 stimuli train, 1/3 held-out)
uv run python -m src.approaches.linear_probing.train_lp --dataset faced --generalization

# Disable mixup or AMP
uv run python -m src.approaches.linear_probing.train_lp --no-mixup --no-amp
```

### Evaluation strategies
1. **10-fold cross-subject CV** (default): KFold(n_splits=10, shuffle=True, seed=42)
2. **Stimulus generalization** (`--generalization`): train on 2/3 stimuli per emotion, validate on held-out 1/3 stimuli + unseen subjects

---

## Fine-Tuning pipeline (LoRA)

Two-stage pipeline faithful to official REVE downstream task:
1. **LP warmup** — frozen encoder, train `cls_query_token` + linear head (same as LP)
2. **LoRA FT** — LoRA adapters on encoder attention + head + query token

### Train

```bash
# All folds, FACED, 9-class (LoRA, default)
uv run python -m src.approaches.fine_tuning.train_ft \
    --dataset faced --task 9-class

# Single fold
uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --fold 1

# Custom LoRA rank
uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --lora-rank 8

# Full fine-tuning (no LoRA — unfreezes entire encoder in stage 2)
uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --fullft

# Official REVE static split: train 0-79, val 80-99, test 100-122 (FACED only)
uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --revesplit
uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --fullft --revesplit

# Generalization evaluation
uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --generalization

# Smoke test
uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --fold 1 --lp-epochs 2 --ft-epochs 3
```

### Evaluation modes (mutually exclusive)
1. **10-fold cross-subject CV** (default): saves `summary_*.json` via `print_fold_summary`
2. **Static split** (`--revesplit`, FACED only): single run, train/val/test fixed by subject ID range, saves `summary_*_revesplit.json`
3. **Stimulus generalization** (`--generalization`): same k-fold subjects but 2/3 stimuli for train, 1/3 held-out

### FT training details

| Setting | LP warmup stage | FT LoRA / full-FT stage |
|---|---|---|
| Optimizer | StableAdamW (betas=0.92/0.999, wd=0.01) | same |
| LR | 5e-3 | 1e-4 |
| LR schedule | Exp warmup (3 ep) + ReduceLROnPlateau | Exp warmup (5 ep) + ReduceLROnPlateau |
| Epochs | 20 max | 200 max |
| Early stopping | patience=10 (val_acc, `>` threshold) | patience=20 |
| Scheduler patience | 6 | 6 |
| Grad clip | max_norm=2.0 | max_norm=2.0 |
| Dropout | 0.05 | 0.1 |
| Batch size | 64 | 64 |
| Precision | AMP float16 | AMP float16 |
| Augmentation | Mixup | Mixup |
| Best model metric | val_acc | val_acc |

### LoRA configuration
- **Targets**: `transformer.layers.{i}.0.to_qkv` and `transformer.layers.{i}.0.to_out` (22 layers)
- **Default**: rank=16, alpha=16, dropout=0.0
- **Library**: peft (`LoraConfig` + `get_peft_model`)

### Saved checkpoints
- LP warmup: saves only trainable params (cls_query_token + head), restores with `strict=False`
- FT stage (LoRA): saves `lora_adapter/` (peft convention) + `head_weights.pt` separately
- FT stage (full FT): saves single `full_model.pt` with complete state dict

### W&B
- Project: `eeg-ft-v2`, Entity: `zl-tudelft-thesis`
- Metrics: `val_acc`, `val_bal_acc`, `val_auroc`, `val_f1`, `val_loss`

---

## SupCon joint training pipeline (CE + Supervised Contrastive Loss)

Joint objective: `L = alpha * L_CE + (1 - alpha) * L_SupCon` where L_SupCon is Khosla et al. 2020 (L_sup_out).

Two-stage pipeline (same as FT):
1. **LP warmup** — frozen encoder, CE only, projection head frozen
2. **FT stage** — LoRA/full FT + joint CE + SupCon loss

### Train

```bash
# All folds, FACED, 9-class (LoRA + SupCon, default)
uv run python -m src.approaches.supcon.train_sc \
    --dataset faced --task 9-class

# Custom SupCon hyperparameters
uv run python -m src.approaches.supcon.train_sc \
    --dataset faced --alpha 0.7 --temperature 0.1

# Different projection head input representation
uv run python -m src.approaches.supcon.train_sc \
    --dataset faced --supcon-repr mean    # or: context (default), both

# Full fine-tuning (no LoRA)
uv run python -m src.approaches.supcon.train_sc --dataset faced --fullft

# Official REVE static split
uv run python -m src.approaches.supcon.train_sc --dataset faced --revesplit

# Generalization evaluation
uv run python -m src.approaches.supcon.train_sc --dataset faced --generalization

# Smoke test
uv run python -m src.approaches.supcon.train_sc --dataset faced --fold 1 --lp-epochs 2 --ft-epochs 3
```

### SupCon-specific settings

| Setting | Default | Description |
|---|---|---|
| `--alpha` | 0.5 | CE weight: `L = alpha*CE + (1-alpha)*SupCon` |
| `--temperature` | 0.07 | SupCon temperature tau |
| `--proj-dim` | 128 | Projection head output dim |
| `--proj-hidden` | 512 | Projection head hidden dim |
| `--supcon-repr` | `context` | Projection input: `context` (query-attention), `mean` (mean-pool), `both` (concat) |

### Architecture

- **Classifier head**: same as FT (RMSNorm → Dropout → Linear)
- **Projection head**: Linear(input_dim, 512) → ReLU → Linear(512, 128) → L2-normalize
- **Projection input** (configurable via `--supcon-repr`):
  - `context`: 512-dim query-attention context vector (default)
  - `mean`: 512-dim mean-pool of all patch tokens
  - `both`: 1024-dim concatenation of context + mean

### SupCon loss (Khosla L_sup_out)

For each anchor i in a batch of B samples:
- **P(i)** = same-class samples (excluding self) — positives
- **A(i)** = all samples except self — denominator
- Loss averages log-ratio over positive pairs per anchor
- Samples with no positives (singleton class) excluded from mean
- Mixup disabled when SupCon active (labels must be clean for pair identification)

### Training details

| Setting | LP warmup stage | FT + SupCon stage |
|---|---|---|
| Loss | CE only | alpha * CE + (1-alpha) * SupCon |
| Optimizer | StableAdamW (betas=0.92/0.999, wd=0.01) | same |
| LR | 5e-3 | 1e-4 |
| Mixup | disabled | disabled |
| Projection head | frozen | trainable |

All other settings (epochs, patience, grad clip, AMP, etc.) match the FT pipeline.

### Saved checkpoints
- Same format as FT (LoRA adapter + head weights, no projection head in main checkpoint)
- Projection head saved separately as `projection_head.pt` for analysis

### W&B
- Project: `eeg-sc-v2`, Entity: `zl-tudelft-thesis`
- Extra metrics: `train/loss_ce`, `train/loss_sc` (component losses)

---

## Official LP training details (official mode)

| Setting | Value |
|---|---|
| Optimizer | StableAdamW (betas=[0.92, 0.999], wd=0.01) |
| LR | 5e-3 |
| LR schedule | Exponential warmup (3 epochs) + ReduceLROnPlateau |
| Epochs | 20 max, patience=10 |
| Batch size | 64 |
| Augmentation | Mixup (λ ~ Beta, blended CE loss) |
| Precision | AMP float16 |
| Grad clip | max_norm=100 |
| Dropout | 0.05 |
| Scale factor | 1000 (µV → mV-range) |

### Pooling modes
- `no`: query attention → concat [query, patches] → flatten → `(B, (1 + C*H)*E)` → head
  - FACED 10s window: output dim = (1 + 32×11)×512 = **180,736**
- `last`: query attention → squeeze → `(B, E=512)` → head
- `last_avg`: mean pool across patches → `(B, E=512)` → head

Head architecture: `RMSNorm → Dropout → Linear(in, n_classes)`

---

## W&B logging

- Project: `eeg-lp-v2`, Entity: `zl-tudelft-thesis`
- Set `USE_WANDB = False` in `src/approaches/linear_probing/config.py` to disable
- Metrics logged: `val_acc`, `val_bal_acc`, `val_auroc`, `val_f1`

---

## Key conventions

- **No MNE** in preprocessing — used `scipy.signal.resample` (matches official REVE preprocessing)
- **scale_factor=1000** applied at dataset load time, NOT at preprocessing save time
- Preprocessed files store raw µV values as float32 .npy
- `reve_official/` is a read-only reference — will be deleted after implementation is complete
- Windows: `NUM_WORKERS=0` (multiprocessing not supported on Windows with DataLoader)
