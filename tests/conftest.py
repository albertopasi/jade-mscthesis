"""
Shared pytest fixtures for the test suite.

Provides:
  - Synthetic .npy data matching FACED / THU-EP preprocessed shapes
  - Synthetic raw .pkl / .mat data for preprocessing pipeline tests
  - Temporary directories pre-populated with fake subject files
  - THU-EP config override that points to temporary paths
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import pytest

# Shape constants (mirroring src code)

FACED_N_STIMULI = 28
FACED_N_CHANNELS = 32
FACED_RAW_TIMEPOINTS = 7500  # 250 Hz × 30 s
FACED_PREP_TIMEPOINTS = 6000  # 200 Hz × 30 s
FACED_RAW_SHAPE = (FACED_N_STIMULI, FACED_N_CHANNELS, FACED_RAW_TIMEPOINTS)
FACED_PREP_SHAPE = (FACED_N_STIMULI, FACED_N_CHANNELS, FACED_PREP_TIMEPOINTS)

THUEP_N_STIMULI = 28
THUEP_N_CHANNELS_RAW = 32
THUEP_N_CHANNELS_PREP = 30
THUEP_N_BANDS = 6
THUEP_RAW_TIMEPOINTS = 7500
THUEP_PREP_TIMEPOINTS = 6000
THUEP_RAW_SHAPE = (THUEP_RAW_TIMEPOINTS, THUEP_N_CHANNELS_RAW, THUEP_N_STIMULI, THUEP_N_BANDS)
THUEP_PREP_SHAPE = (THUEP_N_STIMULI, THUEP_N_CHANNELS_PREP, THUEP_PREP_TIMEPOINTS)


# Helpers


def _make_fake_eeg(shape: tuple, rng: np.random.RandomState | None = None) -> np.ndarray:
    """Generate plausible-looking fake EEG data (small amplitude, float32)."""
    if rng is None:
        rng = np.random.RandomState(0)
    # Simulate µV-range values (±200 µV)
    return (rng.randn(*shape) * 50).astype(np.float32)


# FACED fixtures


@pytest.fixture
def faced_raw_dir(tmp_path: Path) -> Path:
    """Create a temp dir with 3 fake FACED raw .pkl files (sub000-sub002)."""
    raw_dir = tmp_path / "FACED" / "Processed_data"
    raw_dir.mkdir(parents=True)
    rng = np.random.RandomState(42)
    for sid in range(3):
        data = _make_fake_eeg(FACED_RAW_SHAPE, rng)
        with open(raw_dir / f"sub{sid:03d}.pkl", "wb") as f:
            pickle.dump(data, f)
    return raw_dir


@pytest.fixture
def faced_preprocessed_dir(tmp_path: Path) -> Path:
    """Create a temp dir with 5 fake FACED preprocessed .npy files (sub000-sub004)."""
    prep_dir = tmp_path / "FACED" / "preprocessed_v2"
    prep_dir.mkdir(parents=True)
    rng = np.random.RandomState(42)
    for sid in range(5):
        data = _make_fake_eeg(FACED_PREP_SHAPE, rng)
        np.save(prep_dir / f"sub{sid:03d}.npy", data)
    return prep_dir


# THU-EP fixtures


@pytest.fixture
def thuep_raw_dir(tmp_path: Path) -> Path:
    """Create a temp dir with 3 fake THU-EP raw .mat (HDF5) files (sub_1-sub_3)."""
    raw_dir = tmp_path / "thu_ep" / "EEG data"
    raw_dir.mkdir(parents=True)
    rng = np.random.RandomState(42)
    for sid in range(1, 4):
        data = _make_fake_eeg(THUEP_RAW_SHAPE, rng)
        mat_path = raw_dir / f"sub_{sid}.mat"
        with h5py.File(mat_path, "w") as f:
            f.create_dataset("data", data=data)
    return raw_dir


@pytest.fixture
def thuep_preprocessed_dir(tmp_path: Path) -> Path:
    """Create a temp dir with 5 fake THU-EP preprocessed .npy files (sub_01-sub_05)."""
    prep_dir = tmp_path / "thu_ep" / "preprocessed_v2"
    prep_dir.mkdir(parents=True)
    rng = np.random.RandomState(42)
    for sid in range(1, 6):
        data = _make_fake_eeg(THUEP_PREP_SHAPE, rng)
        np.save(prep_dir / f"sub_{sid:02d}.npy", data)
    return prep_dir


# THU-EP config fixture (overrides paths to point at tmp dirs)


@pytest.fixture
def thuep_test_config(tmp_path: Path) -> Dict:
    """Return a dict that can be written to a temporary thu_ep.yml for testing."""
    return {
        "paths": {
            "raw_data_dir": str(tmp_path / "thu_ep" / "EEG data"),
            "ratings_dir": str(tmp_path / "thu_ep" / "Ratings"),
            "others_dir": str(tmp_path / "thu_ep" / "Others"),
            "preprocessed_output_dir": str(tmp_path / "thu_ep" / "preprocessed_v2"),
        },
        "channels": {
            "all_channels": [
                "Fp1",
                "Fp2",
                "Fz",
                "F3",
                "F4",
                "F7",
                "F8",
                "FC1",
                "FC2",
                "FC5",
                "FC6",
                "Cz",
                "C3",
                "C4",
                "T7",
                "T8",
                "A1",
                "A2",
                "CP1",
                "CP2",
                "CP5",
                "CP6",
                "Pz",
                "P3",
                "P4",
                "P7",
                "P8",
                "PO3",
                "PO4",
                "Oz",
                "O1",
                "O2",
            ],
            "channels_to_remove": ["A1", "A2"],
        },
        "bands": {
            "names": ["delta", "theta", "alpha", "beta", "gamma", "broad-band"],
            "extract_band_index": 5,
        },
        "sampling": {
            "original_sfreq_hz": 250.0,
            "target_sfreq_hz": 200.0,
            "original_n_samples": 7500,
            "target_n_samples": 6000,
        },
        "dataset": {
            "n_subjects": 80,
            "n_stimuli": 28,
            "n_channels": 32,
            "n_bands": 6,
            "expected_raw_shape": [7500, 32, 28, 6],
            "expected_preprocessed_shape": [28, 30, 6000],
        },
        "preprocessing": {
            "artifact_threshold_std": 15.0,
            "steps_enabled": {
                "remove_reference_channels": True,
                "extract_band": True,
                "downsample": True,
                "z_normalize": False,
                "artifact_clipping": False,
                "export_npy": True,
            },
        },
        "options": {"verbose": False},
    }
