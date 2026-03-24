# Datasets

PyTorch dataset classes and cross-validation utilities for FACED and THU-EP.

Both datasets share the same 28-stimulus, 9-emotion label structure and use a common base class.

---

## Files

| File | Description |
|---|---|
| `base.py` | `EEGWindowDataset` base class, shared label maps, `build_stimulus_label_map()` |
| `faced_dataset.py` | `FACEDWindowDataset` — 123 subjects, 32 channels |
| `thu_ep_dataset.py` | `THUEPWindowDataset` — 79 subjects (80 minus subject 75), 30 channels |
| `folds.py` | `get_all_subjects()`, `get_kfold_splits()`, `get_stimulus_generalization_split()` |

---

## Label structure

Both datasets use the same 28-stimulus → 9-class mapping:

| Stimuli | Emotion | 9-class label | Binary label |
|---|---|---|---|
| 0–2 | Anger | 0 | negative |
| 3–5 | Disgust | 1 | negative |
| 6–8 | Fear | 2 | negative |
| 9–11 | Sadness | 3 | negative |
| 12–15 | Neutral | 4 | excluded |
| 16–18 | Amusement | 5 | positive |
| 19–21 | Inspiration | 6 | positive |
| 22–24 | Joy | 7 | positive |
| 25–27 | Tenderness | 8 | positive |

In binary mode, Neutral stimuli are dropped (label=None → excluded from dataset).

---

## Dataset specs

### FACED
- 123 subjects (0-indexed: 0–122), no excluded subjects
- 32 channels @ 200 Hz, 28 stimuli × 30s = 6000 timepoints per stimulus
- Preprocessed shape: `(28, 32, 6000)` float32, raw µV
- File pattern: `sub{NNN}.npy`

### THU-EP
- 80 subjects (1-indexed: 1–80), subject 75 excluded
- Per-subject stimulus exclusions: sub37→{15,21,24}, sub46→{3,9,17,23,26}
- 30 channels @ 200 Hz (32 original minus A1, A2)
- Preprocessed shape: `(28, 30, 6000)` float32, raw µV
- File pattern: `sub_{XX}.npy`

Both datasets divide values by `scale_factor=1000` at load time (µV → mV-range).

---

## Windowing

Each 30s stimulus (6000 timepoints) is sliced into non-overlapping 10s windows (2000 timepoints) by default, yielding **3 windows per stimulus**.

```python
from src.datasets.faced_dataset import FACEDWindowDataset

ds = FACEDWindowDataset(
    subject_ids=[0, 1, 2],
    task_mode="9-class",       # "binary" or "9-class"
    data_root=Path("data/FACED/preprocessed_v2"),
    window_size=2000,          # 10s at 200 Hz
    stride=2000,               # non-overlapping
    scale_factor=1000.0,
)
eeg, label = ds[0]  # eeg: (32, 2000) float32 tensor
```

---

## Cross-validation

```python
from src.datasets.folds import get_all_subjects, get_kfold_splits, get_stimulus_generalization_split

subjects = get_all_subjects("faced")          # list of 123 subject IDs
folds = get_kfold_splits(subjects)            # 10-fold CV splits (seed=42)

train_stim, val_stim = get_stimulus_generalization_split("9-class", seed=123)
# train_stim: 2/3 stimuli per emotion group
# val_stim:   1/3 stimuli per emotion group (held-out)
```
