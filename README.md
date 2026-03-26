# 🧠 JADE: Cross-Subject Generalization in EEG Emotion Recognition from EEG Data

*Master Thesis in Computer Science, [TU Delft](https://www.tudelft.nl/en/), conducted at [Zander Labs](https://zanderlabs.com/).*

Welcome to the **JADE** MSc Thesis Project! 🚀 

This repository is dedicated to testing and improving **cross-subject generalization** for emotion recognition using EEG data. We leverage the power of the [REVE Foundation Model](https://brain-bzh.github.io/reve/) (an advanced EEG pre-trained transformer) alongside **Supervised Contrastive Learning (SupCon)** to enhance the extraction of subject-invariant features.

The goal is to move beyond subject-specific constraints and build robust models capable of accurately predicting emotions on entirely unseen subjects.

---

## 🎯 Project Scope

1. **Reproduce & Evaluate:** Implement the Linear Probing (LP) downstream task on the REVE foundation model to establish a solid baseline for cross-subject generalization.
2. **Fine-Tuning:** Fine-tune REVE using LoRA (Low-Rank Adaptation) while keeping the classification head identical to the linear probing phase.
3. **Enhance with SupCon:** Introduce Supervised Contrastive Learning during the fine-tuning process. By attaching a projection head and jointly training with a weighted sum of Cross-Entropy Loss and SupCon Loss, we aim to pull representations of the same emotion closer together across different subjects, fostering true subject-invariance.

---

## 🚀 Getting Started

### 1️⃣ Prerequisites

* **Python Package Manager:** We use `uv` for lightning-fast dependency management. [Install uv here](https://docs.astral.sh/uv/getting-started/installation/).
* **Datasets:** You will need to download the raw data for the two supported datasets:
  * **FACED Dataset:** [Paper Link](https://doi-org.tudelft.idm.oclc.org/10.1038/s41597-023-02650-w) | [Download Link](https://doi-org.tudelft.idm.oclc.org/10.7303/syn50614194)
  * **THU-EP Dataset:** [Paper Link](https://doi.org/10.1016/j.neuroimage.2021.118819) | [Download Link](https://cloud.tsinghua.edu.cn/d/3d176032a5a545c1b927/)

Place the downloaded data in the corresponding folders under `data/FACED/` and `data/thu ep/`.

### 2️⃣ Installation & Setup

Sync the environment using `uv`:

```bash
uv sync
```

### 3️⃣ Download Pre-Trained Models

Download the pre-trained REVE base model, large model, and position embeddings directly from Hugging Face (requires having access to REVE model, and having set up an Access token):

```bash
uv run python -m src.download_reve.download_models
```

### 4️⃣ Data Preprocessing

Preprocess the raw dataset files (converts `.pkl`/`.mat` to `.npy`, handles resampling, and standardizes channels):

```bash
# Preprocess FACED dataset
uv run python -m src.preprocessing.faced.run_preprocessing

# Preprocess THU-EP dataset
uv run python -m src.preprocessing.thu_ep.run_preprocessing
```
*(Tip: You can append `--validate` to the commands above to verify the output shapes and types!)*

---

## Training Pipelines

*Note: Always use `uv run python` to ensure you are executing within the correct isolated environment.*

### 🔍 Linear Probing (LP)

Train the linear classification head while keeping the REVE encoder frozen. You can choose different configurations depending on the task.

**Example: FACED dataset, 9-class emotion recognition, across all folds**
```bash
uv run python -m src.approaches.linear_probing.train_lp --dataset faced --task 9-class
```

**Example: THU-EP dataset, binary task, single fold**
```bash
uv run python -m src.approaches.linear_probing.train_lp --dataset thu-ep --task binary --fold 1
```

**Example: Faced dataset, binary task, custom window size and stride (default: size 10, stride 10 (seconds))**
```bash
uv run python -m src.approaches.linear_probing.train_lp --dataset faced --task binary --window 6 --stride 6
```

### 🎛️ Fine-Tuning (FT) with LoRA

The fine-tuning pipeline is a two-stage process: an LP warmup stage, followed by LoRA adapter training on the encoder attention layers.

**Example: All folds, FACED dataset, 9-class task**
```bash
uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --task 9-class
```

**Example: Single fold evaluation**
```bash
uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --fold 1
```

**Example: Custom LoRA Rank**
```bash
uv run python -m src.approaches.fine_tuning.train_ft --dataset faced --lora-rank 8
```

---

## 📂 Project Structure

Here is a high-level overview of the repository structure to help you navigate:

```text
jade-mscthesis/
├── configs/                # YAML configuration files (e.g., THU-EP paths, channels, sampling)
├── data/                   # Raw and preprocessed datasets
│   ├── FACED/              
│   └── thu ep/             
├── models/                 # Downloaded local models (REVE base, positions, etc.)
├── outputs/                # Checkpoints (LP and FT) saved per fold/run
├── src/                    # Main source code directory
│   ├── approaches/         # Training algorithms
│   │   ├── shared/         # Shared utilities (summaries, metrics)
│   │   ├── linear_probing/ # LP logic, models, and training loops
│   │   └── fine_tuning/    # FT logic, LoRA injection, and training loops
│   ├── datasets/           # PyTorch Dataset classes and cross-validation folding logic
│   ├── preprocessing/      # Scripts to transform raw EEG data to clean .npy tensors
│   ├── exploration/        # Jupyter notebooks / scripts for initial data exploration
│   └── utils/              # General utilities, PyTorch Lightning callbacks
└── src/download_reve/      # Script to download HF REVE models
```

---

## 🔮 Future Work
* Integrating the Supervised Contrastive Learning (SupCon) projection head to the existing LoRA Fine-Tuning pipeline.
* Comprehensive evaluation on stimulus-generalization vs. random-split cross-validation.

