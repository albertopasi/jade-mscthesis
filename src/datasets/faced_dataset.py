"""
faced_dataset.py — FACED EEG dataset loader for downstream tasks.

Provides:
  - FACED_CHANNELS: official 32-channel electrode names.
  - FACEDWindowDataset: raw EEG sliding-window dataset loaded into RAM.

Loads preprocessed .npy files (28, 32, 6000) @ 200 Hz from
data/FACED/preprocessed_v2/. Run src.preprocessing.faced.run_preprocessing
first to generate them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


# 32 FACED electrode names in official order (matches reve_official config)
FACED_CHANNELS: list[str] = [
    "FP1", "FP2", "FZ", "F3", "F4", "F7", "F8",
    "FC1", "FC2", "FC5", "FC6",
    "CZ", "C3", "C4", "T3", "T4",
    "CP1", "CP2", "CP5", "CP6",
    "PZ", "P3", "P4", "T5", "T6",
    "PO3", "PO4", "OZ", "O1", "O2",
    "A2", "A1",
]

N_SUBJECTS        = 123
N_STIMULI         = 28
N_CHANNELS        = 32
TARGET_SFREQ      = 200
STIMULUS_DURATION = 30
N_TIMEPOINTS      = TARGET_SFREQ * STIMULUS_DURATION  # 6000

# 9-class labels per stimulus (official, from preprocessing_faced.py)
_FACED_LABELS: np.ndarray = np.array(
    [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8]
)

# No globally excluded subjects by default
EXCLUDED_SUBJECTS: set[int] = set()
EXCLUDED_STIMULI: Dict[int, set[int]] = {}

# Binary mapping: neg=0, pos=1, neutral=None (dropped)
_BINARY_REMAP = {0: 0, 1: 0, 2: 0, 3: 0, 4: None, 5: 1, 6: 1, 7: 1, 8: 1}


def _build_stimulus_label_map(task_mode: str) -> Dict[int, Optional[int]]:
    label_map: Dict[int, Optional[int]] = {}
    for stim_idx in range(N_STIMULI):
        class_9 = int(_FACED_LABELS[stim_idx])
        if task_mode == "9-class":
            label_map[stim_idx] = class_9
        else:
            label_map[stim_idx] = _BINARY_REMAP[class_9]
    return label_map


class FACEDWindowDataset(Dataset):
    """
    PyTorch Dataset for FACED EEG data, served as sliding windows.

    Loads preprocessed .npy files (28, 32, 6000) into CPU RAM at construction time.
    Run src.preprocessing.faced.run_preprocessing first.

    Args:
        subject_ids:      List of 0-indexed subject IDs to include.
        task_mode:        'binary' or '9-class'.
        data_root:        Path to preprocessed_v2/ directory.
        window_size:      Timepoints per window (default 2000 = 10 s).
        stride:           Stride between windows (default 2000 = non-overlapping).
        scale_factor:     Divide raw EEG by this (default 1000.0, matching official).
        stimulus_filter:  If provided, only include these stimulus indices.
    """

    def __init__(
        self,
        subject_ids: List[int],
        task_mode: str,
        data_root: Path,
        window_size: int = 2000,
        stride: int = 2000,
        scale_factor: float = 1000.0,
        stimulus_filter: Optional[set[int]] = None,
    ) -> None:
        super().__init__()

        assert task_mode in ("binary", "9-class"), (
            f"task_mode must be 'binary' or '9-class', got '{task_mode}'"
        )

        self.task_mode = task_mode
        self.data_root = Path(data_root)
        self.window_size = window_size
        self.stride = stride
        self.scale_factor = scale_factor
        self.stimulus_filter = stimulus_filter

        self._label_map = _build_stimulus_label_map(task_mode)

        # Load all subjects into RAM
        # data_cache[subject_id] = np.ndarray of shape (28, 32, 6000)
        self.data_cache: Dict[int, np.ndarray] = {}

        for sid in subject_ids:
            if sid in EXCLUDED_SUBJECTS:
                continue
            npy_path = self.data_root / f"sub{sid:03d}.npy"
            if not npy_path.exists():
                raise FileNotFoundError(
                    f"Preprocessed file not found: {npy_path}\n"
                    f"Run: uv run python -m src.preprocessing.faced.run_preprocessing"
                )
            self.data_cache[sid] = np.load(npy_path)  # (28, 32, 6000)

        # Build flat index of valid (subject, stimulus, window_start)
        n_windows = (N_TIMEPOINTS - window_size) // stride + 1

        self.index: List[Tuple[int, int, int]] = []

        for sid, data in self.data_cache.items():
            for stim_idx in range(N_STIMULI):
                if sid in EXCLUDED_STIMULI and stim_idx in EXCLUDED_STIMULI[sid]:
                    continue
                label = self._label_map.get(stim_idx)
                if label is None:
                    continue
                if stimulus_filter is not None and stim_idx not in stimulus_filter:
                    continue
                for w in range(n_windows):
                    self.index.append((sid, stim_idx, w * stride))

    def __len__(self) -> int:
        return len(self.index)

    @property
    def labels(self) -> List[int]:
        return [self._label_map[stim_idx] for _, stim_idx, _ in self.index]

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        subject_id, stim_idx, window_start = self.index[idx]
        window = self.data_cache[subject_id][
            stim_idx, :, window_start : window_start + self.window_size
        ]
        eeg_tensor = torch.from_numpy(window) / self.scale_factor
        return eeg_tensor, self._label_map[stim_idx]
