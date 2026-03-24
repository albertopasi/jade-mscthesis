"""
faced_dataset.py — FACED EEG dataset for downstream tasks.

Loads preprocessed .npy files (28, 32, 6000) @ 200 Hz from data/FACED/preprocessed_v2/.
Run src.preprocessing.faced.run_preprocessing first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from .base import EEGWindowDataset


# 32 FACED electrode names in official order
FACED_CHANNELS: list[str] = [
    "FP1", "FP2", "FZ", "F3", "F4", "F7", "F8",
    "FC1", "FC2", "FC5", "FC6",
    "CZ", "C3", "C4", "T3", "T4",
    "CP1", "CP2", "CP5", "CP6",
    "PZ", "P3", "P4", "T5", "T6",
    "PO3", "PO4", "OZ", "O1", "O2",
    "A2", "A1",
]

EXCLUDED_SUBJECTS: set[int] = set()
EXCLUDED_STIMULI: Dict[int, set[int]] = {}


class FACEDWindowDataset(EEGWindowDataset):
    """PyTorch Dataset for FACED EEG data, served as sliding windows.

    Args:
        subject_ids:     List of 0-indexed subject IDs (0–122).
        task_mode:       'binary' or '9-class'.
        data_root:       Path to preprocessed_v2/ directory.
        window_size:     Timepoints per window (default 2000 = 10 s).
        stride:          Stride between windows (default 2000 = non-overlapping).
        scale_factor:    Divide raw EEG by this (default 1000.0).
        stimulus_filter: If provided, only include these stimulus indices.
    """

    EXCLUDED_SUBJECTS = EXCLUDED_SUBJECTS
    EXCLUDED_STIMULI  = EXCLUDED_STIMULI

    def _subject_path(self, sid: int) -> Path:
        return self.data_root / f"sub{sid:03d}.npy"
