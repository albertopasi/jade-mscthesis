"""
thu_ep_dataset.py — THU-EP EEG dataset for downstream tasks.

Loads preprocessed .npy files (28, 30, 6000) @ 200 Hz from data/thu ep/preprocessed_v2/.
Run src.preprocessing.thu_ep.run_preprocessing first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from .base import EEGWindowDataset


# 30 THU-EP electrode names (32 original minus A1, A2)
THU_EP_CHANNELS: list[str] = [
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8",
    "FC1", "FC2", "FC5", "FC6",
    "Cz", "C3", "C4", "T3", "T4",
    "CP1", "CP2", "CP5", "CP6",
    "Pz", "P3", "P4", "T5", "T6",
    "PO3", "PO4", "Oz", "O1", "O2",
]

# Subject 75 is corrupted — skip entirely
# Subject 37: stimuli 15, 21, 24 are corrupted
# Subject 46: stimuli 3, 9, 17, 23, 26 are corrupted
EXCLUDED_SUBJECTS: set[int] = {75}
EXCLUDED_STIMULI: Dict[int, set[int]] = {
    37: {15, 21, 24},
    46: {3, 9, 17, 23, 26},
}


class THUEPWindowDataset(EEGWindowDataset):
    """PyTorch Dataset for THU-EP EEG data, served as sliding windows.

    Args:
        subject_ids:     List of 1-indexed subject IDs (1–80, excluding 75).
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
        return self.data_root / f"sub_{sid:02d}.npy"
