# Reproducing Results

The exact commands to reproduce every headline number, table, and figure in the thesis. Each section lists: (1) the training command, (2) the post-hoc inference command (where applicable), and (3) the output file the result lands in.

All commands assume the working directory is the repo root and the environment is synced (`uv sync`). All commands go through `uv run`.

Walltimes are A100-80GB ballparks. The B=256, 9-class JADE/FT jobs peak around 62 GB GPU RAM; smaller-batch binary jobs are comfortable on 40 GB cards.

---

## 0. One-time setup

```bash
# Environment
uv sync

# REVE weights (one-shot, needs HF token with REVE access)
uv run python -m src.download_reve.download_models

# FACED preprocessing (250 → 200 Hz, ~5 min)
uv run python -m src.preprocessing.faced.run_preprocessing
uv run python -m src.preprocessing.faced.run_preprocessing --validate
```

After this the preprocessed `.npy` files live under [`data/FACED/preprocessed_v2/`](../data/FACED/preprocessed_v2/) and REVE is at [`models/reve_pretrained_original/reve-base/`](../models/reve_pretrained_original/reve-base/).

---

## 1. Main-results table (Table 6.1 in the thesis)

Six runs total — three methods × two tasks. Each run is a full 10-fold cross-subject CV.

### LP — 9-class

```bash
# Train (≈ 4 h on A100-80GB)
sbatch slurm/run_experiment.sh src.approaches.linear_probing.train_lp \
    --dataset faced --task 9-class --no-mixup
```

Per-subject inference:

```bash
uv run python -m src.inference.inference_subject_wise \
    --approach lp --task 9-class \
    --ckpt-root outputs/lp_checkpoints \
    --run-stem lp_faced_v2_9-class_w10s10_pool_no_official_nomixup
```

Output → [`main-results/lp_9-class/lp_faced_v2_9-class_w10s10_pool_no_official_nomixup.json`](../main-results/lp_9-class/) (mean = 50.27 %).

### LP — Binary

```bash
sbatch slurm/run_experiment.sh src.approaches.linear_probing.train_lp \
    --dataset faced --task binary --no-mixup

uv run python -m src.inference.inference_subject_wise \
    --approach lp --task binary \
    --ckpt-root outputs/lp_checkpoints \
    --run-stem lp_faced_v2_binary_w10s10_pool_no_official_nomixup
```

Output → [`main-results/lp_binary/lp_faced_v2_binary_w10s10_pool_no_official_nomixup.json`](../main-results/lp_binary/) (mean = 71.64 %).

### SFT — 9-class

```bash
# Train (≈ 10 h on A100-80GB)
sbatch slurm/run_experiment.sh src.approaches.fine_tuning.train_ft \
    --dataset faced --task 9-class --fullft --no-mixup \
    --batch-size 256 --ft-lr 4e-4

uv run python -m src.inference.inference_subject_wise \
    --approach ft --task 9-class \
    --ckpt-root outputs/ft_checkpoints \
    --run-stem ft_faced_9-class_w10s10_pool_no_r16_b256_lr0.0004_nomixup_fullft
```

Output → [`main-results/ft_9-class/ft_faced_9-class_w10s10_pool_no_r16_b256_lr0.0004_nomixup_fullft.json`](../main-results/ft_9-class/) (mean = 58.52 %).

### SFT — Binary

```bash
sbatch slurm/run_experiment.sh src.approaches.fine_tuning.train_ft \
    --dataset faced --task binary --fullft --no-mixup

uv run python -m src.inference.inference_subject_wise \
    --approach ft --task binary \
    --ckpt-root outputs/ft_checkpoints \
    --run-stem ft_faced_binary_w10s10_pool_no_r16_nomixup_fullft
```

Output → [`main-results/ft_binary/ft_faced_binary_w10s10_pool_no_r16_nomixup_fullft.json`](../main-results/ft_binary/) (mean = 75.52 %). Default config (`batch_size=64, ft_lr=1e-4`).

### JADE — 9-class (winning HPs)

```bash
# Train (≈ 12 h on A100-80GB)
sbatch slurm/run_experiment.sh src.approaches.jade.train_jade \
    --dataset faced --task 9-class --fullft \
    --alpha 0.3 --temperature 0.2 \
    --batch-size 256 --ft-lr 4e-4

uv run python -m src.inference.inference_subject_wise \
    --approach jade --task 9-class \
    --ckpt-root outputs/jade_checkpoints \
    --run-stem jade_faced_9-class_w10s10_pool_no_r16_a0.3_t0.2_context_b256_lr0.0004_fullft
```

Output → [`main-results/jade_9-class/jade_faced_9-class_w10s10_pool_no_r16_a0.3_t0.2_context_b256_lr0.0004_fullft.json`](../main-results/jade_9-class/) (mean = **62.03 %**).

### JADE — Binary (winning HPs)

```bash
# Train (≈ 8 h on A100-80GB)
sbatch slurm/run_experiment.sh src.approaches.jade.train_jade \
    --dataset faced --task binary --fullft \
    --alpha 0.2 --temperature 0.05 \
    --batch-size 128 --ft-lr 1e-4

uv run python -m src.inference.inference_subject_wise \
    --approach jade --task binary \
    --ckpt-root outputs/jade_checkpoints \
    --run-stem jade_faced_binary_w10s10_pool_no_r16_a0.2_t0.05_context_b128_lr0.0001_fullft
```

Output → [`main-results/jade_binary/jade_faced_binary_w10s10_pool_no_r16_a0.2_t0.05_context_b128_lr0.0001_fullft.json`](../main-results/jade_binary/) (mean = **76.32 %**).

---

## 2. Statistical tests (auto-generated docs)

Once the six main-results JSONs are in place:

```bash
# Main results — Wilcoxon + BCa + Holm
uv run python -m src.inference.statistical_tests
# Writes: docs/statistical_tests.md

# Generalization — Friedman + Brown-Forsythe + variance ratios
uv run python -m src.inference.statistical_tests_generalization
# Writes: docs/statistical_tests_generalization.md
```

Both scripts read from `main-results/<approach>_<task>/<stem>.json` (resp. `*_generalization/<stem>_gen_avg.json`). To change which run is treated as the "winner", edit the `RUNS` dict at the top of [`src/inference/statistical_tests.py`](../src/inference/statistical_tests.py).

---

## 3. Figures (per task)

```bash
# All figures for both tasks in one shot
uv run python -m src.visualization.make_all

# Or per task:
uv run python -m src.visualization.make_all --task 9-class
uv run python -m src.visualization.make_all --task binary

# Subject-wise violin (LP / SFT / JADE side by side, both tasks)
uv run python -m src.visualization.plot_violins
```

Outputs:

- `src/visualization/jade_9-class/figures/`
  - `confusion_matrix.pdf` — Figure 6.4 in the thesis
  - `paired_subject_scatter.pdf` — Figure 6.2(b) (JADE vs SFT, 9-class)
  - `paired_subject_scatter_with_lp.pdf` — 2-panel paired scatter
  - `per_class_f1_bars.pdf` / `per_class_f1_bars_with_lp.pdf` — Figure 6.5
  - `subject_histogram.pdf` — Figure 6.1(b)
  - `subject_violin.pdf`
  - `per_class_metrics.csv` / `.tex` — Table 6.3
- `src/visualization/jade_binary/figures/`
  - `confusion_matrix.pdf` — Figure 6.3
  - `paired_subject_scatter.pdf` — Figure 6.2(a) (JADE vs SFT, binary)
  - `paired_subject_scatter_with_lp.pdf`
  - `per_class_f1_bars.pdf`
  - `subject_histogram.pdf` — Figure 6.1(a)
  - `subject_violin.pdf`
  - `per_class_metrics.csv` / `.tex` — Table 6.2

---

## 4. Stimulus-generalization protocol (Table 6.4)

Held-out stimuli on held-out subjects. Each job runs three stimulus-split seeds (`123 456 789`) × 10 folds. Per-fold inference happens in-memory; no checkpoints saved.

> Run via [`slurm/run_faced_generalization.sh`](../slurm/run_faced_generalization.sh) which submits all four jobs (FT 9-class, FT binary, LP 9-class, LP binary). JADE jobs are listed separately in [`slurm/run_gen_smoke.sh`](../slurm/run_gen_smoke.sh).

### JADE generalization

```bash
sbatch slurm/run_experiment.sh src.approaches.jade.train_jade \
    --dataset faced --task 9-class --fullft --generalization \
    --gen-seeds 123 456 789 --no-save-checkpoints \
    --alpha 0.3 --temperature 0.2 \
    --batch-size 256 --ft-lr 4e-4

sbatch slurm/run_experiment.sh src.approaches.jade.train_jade \
    --dataset faced --task binary --fullft --generalization \
    --gen-seeds 123 456 789 --no-save-checkpoints \
    --alpha 0.2 --temperature 0.05 \
    --batch-size 128 --ft-lr 1e-4
```

Each seed produces one summary JSON in `main-results/jade_<task>_generalization/<stem>_gen_s<seed>.json` at the end of its 10 folds — no extra inference step needed.

### Seed averaging

After all seeds finish:

```bash
uv run python -m src.inference.average_gen_seeds --approach jade --task 9-class
uv run python -m src.inference.average_gen_seeds --approach jade --task binary
uv run python -m src.inference.average_gen_seeds --approach ft   --task 9-class
uv run python -m src.inference.average_gen_seeds --approach ft   --task binary
uv run python -m src.inference.average_gen_seeds --approach lp   --task 9-class
uv run python -m src.inference.average_gen_seeds --approach lp   --task binary
```

Each command writes `<common_stem>_gen_avg.json` alongside the per-seed files. Then re-run `statistical_tests_generalization.py` to refresh [`docs/statistical_tests_generalization.md`](statistical_tests_generalization.md).

---

## 5. Hyperparameter sweeps

Not headline results, but documented so the methodology is reproducible.

### Window / stride sweep (LP)

The window and stride were tuned once under LP and frozen for SFT / JADE. The full sweep is in [`slurm/run_lr_sweep.sh`](../slurm/run_lr_sweep.sh) and analyses in [`docs/lp_results_analysis.md`](lp_results_analysis.md).

### `(λ, τ)` sweep at REVE LR (JADE)

```bash
bash slurm/run_jade_grid_completion.sh    # main grid
bash slurm/run_tau_sweep.sh               # τ axis refinement
bash slurm/run_tau_sweep_extra.sh         # additional τ points
```

Results discussion: [`docs/jade_hp_sweep.md`](jade_hp_sweep.md) §3.

### JADE learning-rate sweep (with chosen `(λ, τ)`)

```bash
bash slurm/run_lr_holes.sh                # LR axis at chosen (λ, τ)
bash slurm/run_jade_bulletproof.sh        # bulletproof grid + LR cross-checks
```

Results discussion: [`docs/jade_hp_sweep.md`](jade_hp_sweep.md) §§4–6.

### `(λ, τ)` re-verification at tuned LR

```bash
bash slurm/run_jade_grid_holes.sh
bash slurm/run_jade_binary_b256_extra.sh  # extra binary cells at B=256
```

The chosen configs that came out of this sweep are the ones used in Section 1 above.

---

## 6. Smoke tests (sanity checks while developing)

```bash
# JADE one fold, tiny epochs (~5 min on A100)
uv run python -m src.approaches.jade.train_jade \
    --dataset faced --task 9-class --fullft \
    --fold 1 --lp-epochs 2 --ft-epochs 3 \
    --alpha 0.3 --temperature 0.2 --batch-size 256 --ft-lr 4e-4

# SFT one fold, tiny epochs
uv run python -m src.approaches.fine_tuning.train_ft \
    --dataset faced --task 9-class --fullft --no-mixup \
    --fold 1 --lp-epochs 2 --ft-epochs 3 \
    --batch-size 256 --ft-lr 4e-4

# Generalization smoke (all three pipelines) via slurm
bash slurm/run_gen_smoke.sh
```

The smoke runs exercise the same code paths as the full jobs — if a smoke test passes, the full job will too. Use them before submitting expensive sweeps.

---

## 7. Where the numbers come from (cross-reference)

| Thesis location | File on disk |
|---|---|
| Table 6.1 (Main results) | `main-results/{lp,ft,jade}_{9-class,binary}/*.json` → `subject_wise.mean_acc / std_acc` |
| Table 6.2 (Binary per-class) | `main-results/jade_binary/.../classification_report.per_class` and `src/visualization/jade_binary/figures/per_class_metrics.csv` |
| Table 6.3 (9-class per-class) | `main-results/jade_9-class/.../classification_report.per_class` and `src/visualization/jade_9-class/figures/per_class_metrics.csv` |
| Table 6.4 (Stimulus generalization) | `main-results/*_generalization/*_gen_avg.json` → `subject_wise.mean_acc / std_acc` |
| Table 6.5 (Selected HPs) | [`docs/jade_hp_sweep.md`](jade_hp_sweep.md) §6 + the per-approach `config.py` defaults |
| Figure 6.1 (Subject histograms) | `src/visualization/jade_{9-class,binary}/figures/subject_histogram.pdf` |
| Figure 6.2 (Paired scatter) | `src/visualization/jade_{9-class,binary}/figures/paired_subject_scatter.pdf` |
| Figure 6.3 (Binary CM) | `src/visualization/jade_binary/figures/confusion_matrix.pdf` |
| Figure 6.4 (9-class CM) | `src/visualization/jade_9-class/figures/confusion_matrix.pdf` |
| Figure 6.5 (Per-class F1) | `src/visualization/jade_9-class/figures/per_class_f1_bars_with_lp.pdf` |

Any number that doesn't match between the thesis and the JSON on disk means the JSON was regenerated after the thesis was written — re-run the figure / table commands to refresh.
