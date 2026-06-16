# 🧠 JADE: Cross-Subject Generalization in EEG Emotion Recognition

*Master Thesis in Computer Science, [TU Delft](https://www.tudelft.nl/en/), conducted at [Zander Labs](https://zanderlabs.com/).*

Welcome to the **JADE** MSc Thesis Project! 🚀

This repository tackles **cross-subject generalization** for EEG emotion recognition. The core idea: take a large pretrained EEG foundation model ([REVE](https://brain-bzh.github.io/reve/)) and fine-tune it with a **Supervised Contrastive (SupCon)** objective that explicitly pulls representations of the same emotion together *across different subjects* — so the model works on a brand-new person with no calibration.

Two questions drive the work:

- **RQ1.** Are frozen embeddings from a large pretrained EEG foundation model already sufficient to overcome inter-subject variability — i.e. can REVE serve as a zero-calibration baseline?
- **RQ2.** Does fine-tuning REVE with a supervised contrastive objective (JADE) push generalization further than standard fine-tuning and beyond existing cross-subject contrastive frameworks?

---

## 🎯 Project Scope

Three training strategies are implemented and compared under strict 10-fold cross-subject CV on FACED (123 subjects, 28 video stimuli, 9 emotion categories):

1. **Linear Probing (LP).** REVE frozen, single linear classifier on top. Establishes the lower bound from frozen foundation-model embeddings (answers RQ1).
2. **Supervised Fine-Tuning (SFT).** Two-stage: LP warmup, then full encoder fine-tune with cross-entropy. Baseline for the contribution of explicit alignment.
3. **Joint Alignment and Discriminative Embedding (JADE).** *The proposed method.* Same staged schedule as SFT, but Stage 2 adds a parallel projection head and a SupCon term to the loss: `L = λ · L_CE + (1 − λ) · L_SupCon`. Answers RQ2.

---

## 📊 Results

Cross-subject classification accuracy on FACED, 10-fold cross-subject protocol (best per task in **bold**):

| Method       | Binary acc (%) | Binary std | Nine-class acc (%) | Nine-class std |
|--------------|---------------:|-----------:|-------------------:|---------------:|
| DE+SVM       |          69.50 |      16.00 |              34.90 |          10.70 |
| DE+MLP       |          70.20 |      11.20 |              35.10 |          10.30 |
| DANN         |          50.50 |       7.10 |              54.10 |           8.30 |
| CLISA        |          70.10 |      15.80 |              41.30 |          12.70 |
| CL-CS        |          72.50 |      15.30 |              43.40 |          13.70 |
| DAEST        |          75.40 |       5.50 |              59.30 |           7.70 |
| **LP**       |          71.64 |       9.29 |              50.27 |          13.69 |
| **SFT**      |          75.52 |       7.76 |              58.52 |          14.01 |
| **JADE**     |       **76.32** |    **8.12** |          **62.03** |      **14.70** |


**What this says.**

- **LP already beats CL-CS** (the prior SOTA contrastive cross-subject framework) on 9-class by **+6.87 pp**, and matches it on binary — *without exposing any part of the encoder to FACED during training*. Subject-invariance can be inherited from large-scale pretraining, not learned from one small dataset.
- **LP → SFT → JADE is monotone** on both tasks. On 9-class JADE adds **+3.51 pp over SFT** (Wilcoxon Holm-adjusted *p* = 4.4e−10; 95 % bootstrap CI on the paired mean Δ: [+2.60, +4.44] pp). On binary the same recipe adds **+0.80 pp** and does not reach significance under multiple-comparison correction (*p* = 0.080) — the task is too coarse for label-aware contrastive alignment to help much.
- **JADE establishes a new SOTA on FACED**: +3.82 pp over CL-CS on binary and +18.63 pp on 9-class. Out of 123 held-out subjects, 0 fall below chance on binary and only 1 falls below chance on 9-class — generalization is essentially uniform across the cohort.

### 🎬 Stimulus-generalization — what the model actually learned

We also held out whole video clips, not just subjects, to test whether the model learned emotion or just clip identity.

| Method | Binary acc (%) | 9-class acc (%) |
|---|---:|---:|
| LP | 58.46 | 15.83 |
| SFT | 59.86 | 15.27 |
| **JADE** | **59.55** | **15.88** |
| *chance* | *50.00* | *11.11* |

On **9-class, accuracy collapses to almost chance** (~16 % vs 11 %), all three methods together. Under the standard protocol the same clips are shown to train and test subjects, so a clip-specific signal already scores well; removing the shortcut reveals that most of what looked like emotion recognition was clip recognition. SupCon cannot fix this: with only ~3 clips per emotion (~2 after hold-out), there is no cross-clip variation for the loss to abstract over — and the collapse hits LP too, so the bottleneck is the dataset, not the loss. **Binary survives** (~59 %) because each label spans four emotions and many more clips, averaging out the clip-specific signal.

This is the main limitation of the work and motivates a larger, clip-diverse benchmark.

### Subject-wise behaviour

JADE outperforms SFT on 88/123 subjects on 9-class (mean Δ = +3.51 pp, broad-based across the cohort) and on 61/123 subjects on binary (mean Δ = +0.80 pp, essentially interchangeable). See the auto-generated paired scatter plots and confusion matrices in [`src/visualization/jade_9-class/figures/`](src/visualization/jade_9-class/figures/) and [`src/visualization/jade_binary/figures/`](src/visualization/jade_binary/figures/).

Headline figures:

- Paired per-subject scatter (JADE vs SFT): [`src/visualization/jade_9-class/figures/paired_subject_scatter.pdf`](src/visualization/jade_9-class/figures/paired_subject_scatter.pdf)
- Per-class F1 bars: [`src/visualization/jade_9-class/figures/per_class_f1_bars_with_lp.pdf`](src/visualization/jade_9-class/figures/per_class_f1_bars_with_lp.pdf)
- Confusion matrices: [`src/visualization/jade_9-class/figures/confusion_matrix.pdf`](src/visualization/jade_9-class/figures/confusion_matrix.pdf), [`src/visualization/jade_binary/figures/confusion_matrix.pdf`](src/visualization/jade_binary/figures/confusion_matrix.pdf)

Full paired significance tests (Wilcoxon + BCa + Holm) and per-class breakdowns: [`docs/results_brief.md`](docs/results_brief.md) and [`docs/statistical_tests.md`](docs/statistical_tests.md).

---

## 🚀 Getting Started

### 1️⃣ Prerequisites

* **Python Package Manager:** We use `uv` for lightning-fast dependency management. [Install uv here](https://docs.astral.sh/uv/getting-started/installation/).
* **Dataset:** Download the raw FACED data:
  * **FACED Dataset:** [Paper Link](https://doi.org/10.1038/s41597-023-02650-w) | [Download Link](https://doi.org/10.7303/syn50614194)

Place the downloaded data in `data/FACED/`.

### 2️⃣ Installation & Setup

Sync the environment using `uv`:

```bash
uv sync
```

### 3️⃣ Download Pre-Trained Models

Download the pre-trained REVE base model and position embeddings directly from Hugging Face (requires having access to REVE and an Access token):

```bash
uv run python -m src.download_reve.download_models
```

### 4️⃣ Data Preprocessing

Preprocess the raw FACED files (`.pkl` → `.npy`, FFT-based resampling 250 → 200 Hz, channel standardisation):

```bash
uv run python -m src.preprocessing.faced.run_preprocessing
```

*(Tip: append `--validate` to verify the output shapes and types.)*

---

## 🏋️‍♂️ Training Pipelines

*Note: Always use `uv run python` to ensure you are executing within the correct isolated environment.*

### 🔍 Linear Probing (LP)

Train the linear classification head while keeping the REVE encoder frozen.

```bash
# 9-class, all folds
uv run python -m src.approaches.linear_probing.train_lp --dataset faced --task 9-class

# Binary, single fold
uv run python -m src.approaches.linear_probing.train_lp --dataset faced --task binary --fold 1

# Custom window/stride (default: 10 s window, 10 s stride)
uv run python -m src.approaches.linear_probing.train_lp --dataset faced --task binary --window 6 --stride 6
```

### 🎛️ Supervised Fine-Tuning (SFT)

Two-stage pipeline: LP warmup (frozen encoder, trains query token + head), then encoder fine-tuning. LoRA by default; pass `--fullft` for full fine-tuning.

```bash
# LoRA fine-tuning, 9-class, all folds
uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --task 9-class

# Full fine-tuning (no LoRA)
uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --fullft

# Custom LoRA rank
uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --lora-rank 8
```

### 🧬 JADE (Joint Alignment and Discriminative Embedding)

Same two-stage schedule as SFT, but Stage 2 adds a SupCon term to the loss via a parallel projection head.

```bash
# JADE 9-class, full FT, winning HPs (Wilcoxon Holm-adjusted p = 4.4e-10 vs SFT)
uv run python -m src.approaches.jade.train_jade \
    --dataset faced --task 9-class --fullft \
    --alpha 0.3 --temperature 0.2 \
    --batch-size 256 --ft-lr 4e-4

# JADE binary, full FT, winning HPs
uv run python -m src.approaches.jade.train_jade \
    --dataset faced --task binary --fullft \
    --alpha 0.2 --temperature 0.05 \
    --batch-size 128 --ft-lr 1e-4

# Smoke test: one fold, tiny epochs (~5 min on A100)
uv run python -m src.approaches.jade.train_jade \
    --dataset faced --task 9-class --fullft \
    --fold 1 --lp-epochs 2 --ft-epochs 3 \
    --alpha 0.3 --temperature 0.2 --batch-size 256 --ft-lr 4e-4
```

---

## 📂 Project Structure

```text
jade-mscthesis/
├── configs/                # YAML configs (sampling, channel layouts, etc.)
├── data/                   # Raw + preprocessed FACED
├── models/                 # Downloaded REVE weights and position embeddings
├── outputs/                # Per-fold checkpoints (gitignored)
├── main-results/           # Canonical per-run JSON+NPZ artifacts (one per winning config)
├── src/
│   ├── approaches/         # Training algorithms
│   │   ├── shared/         # Shared utilities (config, dataset, metrics, optimizer, REVE loader)
│   │   ├── linear_probing/ # LP — frozen encoder + linear head
│   │   ├── fine_tuning/    # SFT — LP warmup + full FT (or LoRA)
│   │   └── jade/           # JADE — SFT + joint CE+SupCon
│   ├── datasets/           # FACED window dataset + k-fold splits
│   ├── preprocessing/      # Raw EEG → .npy tensors
│   ├── inference/          # Post-hoc inference, statistical tests, seed averaging
│   ├── visualization/      # All figure-generation scripts + per-task figure folders
│   ├── exploration/        # One-off data exploration scripts
│   └── download_reve/      # HF download for REVE weights
├── slurm/                  # Every sbatch script used to produce the results
├── docs/                   # Findings, methodology, statistical tests — see docs/README.md
└── tests/                  # Fast unit tests (no model, no real data)
```

---

## 📚 Documentation

The `docs/` folder is the source of truth for everything beyond this README:

- [`docs/results_brief.md`](docs/results_brief.md) — detailed results: headline accuracies, per-class metrics, confusion matrices, subject-wise breakdowns
- [`docs/statistical_tests.md`](docs/statistical_tests.md) — paired Wilcoxon + BCa CIs + Holm on main results (auto-generated)
- [`docs/statistical_tests_generalization.md`](docs/statistical_tests_generalization.md) — Friedman + Brown-Forsythe on stimulus-generalization (auto-generated)
- [`docs/jade_approach_design.md`](docs/jade_approach_design.md) — JADE design rationale (loss, projection head, schedule)
- [`docs/jade_hp_sweep.md`](docs/jade_hp_sweep.md) — staged HP tuning protocol, sweep tables, and selected per-task configurations
- [`docs/runs_inventory.md`](docs/runs_inventory.md) — inventory of every run + which sweep cell it belongs to

---

## 📝 Citation

```bibtex
@mastersthesis{pasinato2026jade,
  author = {Alberto Pasinato},
  title  = {Supervised Contrastive Fine-Tuning of EEG Foundation Models for Cross-Subject Emotion Recognition},
  school = {Delft University of Technology},
  year   = {2026},
  type   = {MSc thesis, Computer Science},
  note   = {External project conducted at Zander Labs},
  url    = {http://repository.tudelft.nl/}
}
```

**Contact:** Alberto Pasinato — A.Pasinato@student.tudelft.nl
