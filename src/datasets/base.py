"""
base.py — Shared EEG dataset base class and label utilities.

Both FACED and THU-EP share the same 28-stimulus, 9-emotion structure and
identical sliding-window logic. Only the filename format and excluded
subjects/stimuli differ per dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


# ── Shared label structure (identical for FACED and THU-EP) ──────────────────

N_STIMULI    = 28
N_TIMEPOINTS = 6000  # 30 s at 200 Hz after preprocessing

# 9-class label per stimulus index (0-27)
STIMULUS_LABELS: np.ndarray = np.array(
    [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8]
)

# Emotion names for reference
EMOTION_NAMES = {
    0: "Anger", 1: "Disgust", 2: "Fear", 3: "Sadness", 4: "Neutral",
    5: "Amusement", 6: "Inspiration", 7: "Joy", 8: "Tenderness",
}

# Binary mapping: neg=0, pos=1, neutral=None (dropped)
BINARY_REMAP: Dict[int, Optional[int]] = {
    0: 0, 1: 0, 2: 0, 3: 0,   # negative
    4: None,                    # neutral — dropped
    5: 1, 6: 1, 7: 1, 8: 1,   # positive
}


def build_stimulus_label_map(task_mode: str) -> Dict[int, Optional[int]]:
    """Map each stimulus index (0-27) to its integer label.

    For 'binary': Neutral stimuli (12-15) map to None and are dropped.
    For '9-class': all stimuli get a label 0-8.
    """
    assert task_mode in ("binary", "9-class"), (
        f"task_mode must be 'binary' or '9-class', got '{task_mode}'"
    )
    label_map: Dict[int, Optional[int]] = {}
    for stim_idx in range(N_STIMULI):
        class_9 = int(STIMULUS_LABELS[stim_idx])
        label_map[stim_idx] = class_9 if task_mode == "9-class" else BINARY_REMAP[class_9]
    return label_map


# ── Base dataset class ────────────────────────────────────────────────────────

class EEGWindowDataset(Dataset):
    """
    Base class for FACED and THU-EP EEG datasets.

    Loads all subject .npy files (28, C, 6000) into CPU RAM at construction,
    then serves sliding windows of shape (C, window_size).

    Subclasses must define:
      - EXCLUDED_SUBJECTS: set of subject IDs to skip entirely
      - EXCLUDED_STIMULI:  dict of {subject_id: set of stimulus indices} to skip
      - _subject_path(sid): returns the Path to sub's .npy file
    """

    EXCLUDED_SUBJECTS: set[int] = set()
    EXCLUDED_STIMULI: Dict[int, set[int]] = {}

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

        self.task_mode      = task_mode
        self.data_root      = Path(data_root)
        self.window_size    = window_size
        self.stride         = stride
        self.scale_factor   = scale_factor
        self.stimulus_filter = stimulus_filter

        self._label_map = build_stimulus_label_map(task_mode)

        # Load all subjects into RAM: {sid: ndarray (28, C, 6000)}
        self.data_cache: Dict[int, np.ndarray] = {}
        for sid in subject_ids:
            if sid in self.EXCLUDED_SUBJECTS:
                continue
            path = self._subject_path(sid)
            if not path.exists():
                raise FileNotFoundError(
                    f"Preprocessed file not found: {path}\n"
                    f"Run the preprocessing script first."
                )
            self.data_cache[sid] = np.load(path)

        # Build flat index of all valid (subject, stimulus, window_start)
        n_windows = (N_TIMEPOINTS - window_size) // stride + 1
        self.index: List[Tuple[int, int, int]] = []

        for sid, data in self.data_cache.items():
            for stim_idx in range(data.shape[0]):
                if sid in self.EXCLUDED_STIMULI and stim_idx in self.EXCLUDED_STIMULI[sid]:
                    continue
                if self._label_map.get(stim_idx) is None:
                    continue
                if stimulus_filter is not None and stim_idx not in stimulus_filter:
                    continue
                for w in range(n_windows):
                    self.index.append((sid, stim_idx, w * stride))

    def _subject_path(self, sid: int) -> Path:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.index)

    @property
    def labels(self) -> List[int]:
        """Label for every window in index order (useful for samplers)."""
        return [self._label_map[stim_idx] for _, stim_idx, _ in self.index]

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        sid, stim_idx, window_start = self.index[idx]
        window = self.data_cache[sid][stim_idx, :, window_start : window_start + self.window_size]
        return torch.from_numpy(window.astype(np.float32)) / self.scale_factor, self._label_map[stim_idx]
