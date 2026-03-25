# HPC Data & Model Layout

This document describes which large files live outside the git repo and where
to place them on the HPC machine, so that all code paths resolve correctly
without any changes to source files.

---

## HPC directory layout

The HPC has two mount points:

```
/scratch/apasinato/    ← large files (SSD, fast I/O): data/ and models/
/home/apasinato/       ← rest of the repo (code, configs, outputs, etc.)
```

The git repository root (`jade-mscthesis/`) lives under `/home/apasinato/`.
The `data/` and `models/` directories are NOT committed to git and are be
placed separately on the scratch disk. The code references them via paths in
`configs/thu_ep.yml` and inline constants — see the "path mapping" section
below.

---

## What to copy to `/scratch/apasinato/`

### 1. THU-EP raw data — `data/thu ep/EEG data/`

- **Size**: ~24 GB
- **Contents**: 80 MATLAB v7.3 (HDF5) files, named `sub_1.mat` … `sub_80.mat`
  (1-indexed, no leading zero).
- **Format**: Each file contains a single variable `data` of shape
  `(7500, 32, 28, 6)` = (timepoints at 250 Hz, channels, stimuli, frequency
  bands). dtype float64. Units: **microvolts (µV)**.
- **Why needed**: The preprocessing pipeline reads these to produce
  `preprocessed_v2/`. Once preprocessing is done on the HPC you do NOT need
  to keep them on the fast disk, but they are required for the first run.

### 2. THU-EP auxiliary files — `data/thu ep/Others/` and `data/thu ep/Ratings/`

- **Size**: negligible (two small .mat files)
- `Others/label.mat` — stimulus-to-label mapping (used historically; labels
  are now hard-coded in `src/thu_ep/dataset.py`)
- `Ratings/ratings.mat` — subjective emotion ratings per subject/stimulus

### 3. THU-EP readme — `data/thu ep/Readme.pdf`

- One PDF, included with the dataset.

### 4. FACED preprocessed data — `data/FACED/Processed_data/`

- **Size**: ~6.2 GB
- **Contents**: 123 pickle files named `sub000.pkl` … `sub122.pkl`
  (0-indexed, three-digit zero-padded).
- **Format**: Each file deserialises to a `numpy.ndarray` of shape
  `(28, 32, 7500)` = (stimuli, channels, timepoints at 250 Hz). dtype float64.
  Units: **microvolts (µV)**. Mean ≈ 0 (already re-referenced & bandpass
  filtered by FACED authors).
- **Why needed**: Used by the REVE official reproduction pipeline
  (`reve_eeg-main/`). The preprocessing script reads these and converts them
  to an LMDB database (`data/FACED/processed/`) before training.
- **Train/val/test split** used by REVE official code (fixed, not CV):
  - train: sub000 – sub079 (80 subjects)
  - val:   sub080 – sub099 (20 subjects)
  - test:  sub100 – sub122 (23 subjects)

### 5. FACED metadata — `data/FACED/` (root-level files)

- **Size**: negligible
- `Dataset_description.md`, `Readme.md`, `manifest.csv`,
  `Recording_info.csv`, `Electrode_Location.xlsx`, `DataStructureOfBehaviouralData.xlsx`,
  `Stimuli_info.xlsx`, `Task_event.xlsx`
- Not required at runtime, useful for reference.

### 6. REVE pretrained models — `models/reve_pretrained_original/`

- **Size**: ~1.8 GB total
- These are **local HuggingFace model snapshots** downloaded from
  `brain-bzh/reve-base`, `brain-bzh/reve-large`, and `brain-bzh/reve-positions`.
  They are loaded via `AutoModel.from_pretrained(local_path, trust_remote_code=True)`.

```
models/
└── reve_pretrained_original/
    ├── reve-base/              ← 264 MB   REVE-Base encoder weights
    │   ├── model.safetensors
    │   ├── config.json
    │   ├── configuration_reve.py
    │   ├── modeling_reve.py
    │   └── README.md
    ├── reve-large/             ← 1.5 GB   REVE-Large encoder weights
    │   ├── model.safetensors
    │   ├── config.json
    │   ├── configuration_reve.py
    │   ├── modeling_reve.py
    │   └── README.md
    └── reve-positions/         ← ~8 KB    Electrode position bank
        ├── model.safetensors
        ├── config.json
        ├── configuration_bank.py
        ├── position_bank.py
        ├── positions.json
        └── README.md
```

**Only `reve-base` and `reve-positions` are used** by the thesis code.
`reve-large` is present locally but not referenced anywhere in `src/`.

---

## Directories NOT to copy (ignorable)

- `data/SEED/` — different dataset, unrelated to this thesis
- `data/Zander Labs/` — different dataset, unrelated to this thesis
- `data/thu ep/cl_cs_preprocessed/` — obsolete intermediate format from an
  abandoned experiment, can be deleted entirely
- `data/thu ep/embeddings/` — pre-computed embedding caches from old pipeline
  (z-scored data); will be regenerated after re-preprocessing
- `data/thu ep/preprocessed/` — old preprocessed .npy files with z-score
  normalization applied; **superseded by `preprocessed_v2/`** (see below)
- `outputs/` — training checkpoints and results from old runs; regenerated on HPC

---

## Directories generated on HPC (do not need to transfer)

These are produced by running the pipeline on the HPC and should be created
there from scratch:

| Directory | Created by | Contents |
|-----------|-----------|---------|
| `data/thu ep/preprocessed_v2/` | `run_preprocessing.py` | 79 `.npy` files, shape (28, 30, 6000), float32, raw µV (no z-score) |
| `data/FACED/processed/` | `reve_eeg-main/preprocessing/preprocessing_faced.py` | LMDB database for REVE official training |
| `data/thu ep/embeddings/` | `train_lp.py` (first run) | Cached REVE embeddings per subject |
| `outputs/` | all training scripts | Checkpoints, results, WandB logs |

---

## Path configuration

The code resolves data paths through `configs/thu_ep.yml`. On the HPC, update
the `paths` section to point to the scratch disk:

```yaml
paths:
  raw_data_dir:           "../../scratch/apasinato/data/thu ep/EEG data"
  preprocessed_output_dir: "../../scratch/apasinato/data/thu ep/preprocessed_v2"
```

Or, more robustly, use absolute paths:

```yaml
paths:
  raw_data_dir:           "/scratch/apasinato/data/thu ep/EEG data"
  preprocessed_output_dir: "/scratch/apasinato/data/thu ep/preprocessed_v2"
```

The REVE model path is passed inline when constructing `EmbeddingExtractor`
in `src/approaches/linear_probing/model.py` — update it to:

```python
reve_model_path = Path("/scratch/apasinato/models/reve_pretrained_original/reve-base")
reve_pos_path   = Path("/scratch/apasinato/models/reve_pretrained_original/reve-positions")
```

For the REVE official reproduction (`reve_eeg-main/`), pass `data_root` as a
CLI override:

```bash
python dt.py task=faced data_root=/scratch/apasinato/data ...
```

---

## Summary: what to `rsync` to `/scratch/apasinato/`

```bash
rsync -av --progress \
  "data/thu ep/EEG data/"          /scratch/apasinato/data/thu\ ep/EEG\ data/
  "data/thu ep/Others/"            /scratch/apasinato/data/thu\ ep/Others/
  "data/thu ep/Ratings/"           /scratch/apasinato/data/thu\ ep/Ratings/
  "data/thu ep/Readme.pdf"         /scratch/apasinato/data/thu\ ep/
  "data/FACED/"                    /scratch/apasinato/data/FACED/
  "models/reve_pretrained_original/" /scratch/apasinato/models/reve_pretrained_original/
```

Total to transfer: ≈ **32 GB** (24 GB THU-EP raw + 6.2 GB FACED + 1.8 GB models).
