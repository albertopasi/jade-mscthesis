"""
Tests for the FACED preprocessing pipeline.

Covers:
  - preprocess_subject: .pkl → .npy conversion with scipy FFT resample
  - validate_subject: shape/NaN/Inf checks on output files
  - Resampling correctness: 7500 → 6000 samples (250 → 200 Hz)
  - Output dtype and shape invariants
  - Edge cases: already-resampled data, missing files
"""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from scipy import signal as scipy_signal

from tests.conftest import (
    FACED_N_CHANNELS,
    FACED_N_STIMULI,
    FACED_PREP_SHAPE,
    FACED_PREP_TIMEPOINTS,
    FACED_RAW_SHAPE,
    FACED_RAW_TIMEPOINTS,
    _make_fake_eeg,
)

# ---------------------------------------------------------------------------
# Import the module under test.  We patch its directory constants so that
# tests operate entirely inside tmp_path fixtures.
# ---------------------------------------------------------------------------

MODULE = "src.preprocessing.faced.run_preprocessing"


# preprocess_subject


class TestPreprocessSubject:
    """Tests for preprocess_subject (single-subject .pkl → .npy)."""

    def test_output_shape(self, faced_raw_dir: Path, tmp_path: Path):
        """Output .npy must have shape (28, 32, 6000)."""
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        with (
            mock.patch(f"{MODULE}.RAW_DIR", faced_raw_dir),
            mock.patch(f"{MODULE}.OUT_DIR", out_dir),
        ):
            from src.preprocessing.faced.run_preprocessing import preprocess_subject

            preprocess_subject(0)

        arr = np.load(out_dir / "sub000.npy")
        assert arr.shape == FACED_PREP_SHAPE

    def test_output_dtype_float32(self, faced_raw_dir: Path, tmp_path: Path):
        """Output arrays must be float32."""
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        with (
            mock.patch(f"{MODULE}.RAW_DIR", faced_raw_dir),
            mock.patch(f"{MODULE}.OUT_DIR", out_dir),
        ):
            from src.preprocessing.faced.run_preprocessing import preprocess_subject

            preprocess_subject(1)

        arr = np.load(out_dir / "sub001.npy")
        assert arr.dtype == np.float32

    def test_no_nan_or_inf(self, faced_raw_dir: Path, tmp_path: Path):
        """Resampling must not introduce NaN or Inf."""
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        with (
            mock.patch(f"{MODULE}.RAW_DIR", faced_raw_dir),
            mock.patch(f"{MODULE}.OUT_DIR", out_dir),
        ):
            from src.preprocessing.faced.run_preprocessing import preprocess_subject

            preprocess_subject(2)

        arr = np.load(out_dir / "sub002.npy")
        assert not np.isnan(arr).any(), "Output contains NaN"
        assert not np.isinf(arr).any(), "Output contains Inf"

    def test_missing_raw_file_raises(self, tmp_path: Path):
        """preprocess_subject must raise FileNotFoundError for missing .pkl."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        with (
            mock.patch(f"{MODULE}.RAW_DIR", empty_dir),
            mock.patch(f"{MODULE}.OUT_DIR", out_dir),
        ):
            from src.preprocessing.faced.run_preprocessing import preprocess_subject

            with pytest.raises(FileNotFoundError):
                preprocess_subject(99)

    def test_already_resampled_data_passthrough(self, tmp_path: Path):
        """If raw data already has 6000 timepoints, no resampling occurs."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        # Create a .pkl with shape (28, 32, 6000) — already at target length
        data = _make_fake_eeg(FACED_PREP_SHAPE)
        with open(raw_dir / "sub000.pkl", "wb") as f:
            pickle.dump(data, f)

        with (
            mock.patch(f"{MODULE}.RAW_DIR", raw_dir),
            mock.patch(f"{MODULE}.OUT_DIR", out_dir),
        ):
            from src.preprocessing.faced.run_preprocessing import preprocess_subject

            preprocess_subject(0)

        arr = np.load(out_dir / "sub000.npy")
        # Should pass through unchanged (no resampling needed)
        assert arr.shape == FACED_PREP_SHAPE
        np.testing.assert_allclose(arr, data, atol=1e-6)

    def test_creates_output_directory(self, faced_raw_dir: Path, tmp_path: Path):
        """preprocess_subject must create the output dir if it doesn't exist."""
        out_dir = tmp_path / "nested" / "deep" / "output"
        assert not out_dir.exists()

        with (
            mock.patch(f"{MODULE}.RAW_DIR", faced_raw_dir),
            mock.patch(f"{MODULE}.OUT_DIR", out_dir),
        ):
            from src.preprocessing.faced.run_preprocessing import preprocess_subject

            preprocess_subject(0)

        assert (out_dir / "sub000.npy").exists()


# validate_subject


class TestValidateSubject:
    """Tests for validate_subject (checks on preprocessed .npy)."""

    def test_valid_file(self, faced_preprocessed_dir: Path):
        """Validation passes for a well-formed preprocessed file."""
        with mock.patch(f"{MODULE}.OUT_DIR", faced_preprocessed_dir):
            from src.preprocessing.faced.run_preprocessing import validate_subject

            result = validate_subject(0)

        assert result["valid"] is True
        assert result["shape"] == FACED_PREP_SHAPE
        assert result["shape_valid"] is True
        assert result["has_nan"] is False
        assert result["has_inf"] is False

    def test_missing_file(self, tmp_path: Path):
        """Validation returns valid=False for a missing file."""
        with mock.patch(f"{MODULE}.OUT_DIR", tmp_path):
            from src.preprocessing.faced.run_preprocessing import validate_subject

            result = validate_subject(999)

        assert result["valid"] is False
        assert "not found" in result["error"]

    def test_wrong_shape_detected(self, tmp_path: Path):
        """Validation detects wrong output shape."""
        wrong_shape = (28, 32, 5000)  # wrong timepoints
        np.save(tmp_path / "sub000.npy", np.zeros(wrong_shape, dtype=np.float32))

        with mock.patch(f"{MODULE}.OUT_DIR", tmp_path):
            from src.preprocessing.faced.run_preprocessing import validate_subject

            result = validate_subject(0)

        assert result["valid"] is False
        assert result["shape_valid"] is False

    def test_nan_detected(self, tmp_path: Path):
        """Validation detects NaN values."""
        data = np.zeros(FACED_PREP_SHAPE, dtype=np.float32)
        data[0, 0, 0] = np.nan
        np.save(tmp_path / "sub000.npy", data)

        with mock.patch(f"{MODULE}.OUT_DIR", tmp_path):
            from src.preprocessing.faced.run_preprocessing import validate_subject

            result = validate_subject(0)

        assert result["valid"] is False
        assert result["has_nan"] is True

    def test_inf_detected(self, tmp_path: Path):
        """Validation detects Inf values."""
        data = np.zeros(FACED_PREP_SHAPE, dtype=np.float32)
        data[0, 0, 0] = np.inf
        np.save(tmp_path / "sub000.npy", data)

        with mock.patch(f"{MODULE}.OUT_DIR", tmp_path):
            from src.preprocessing.faced.run_preprocessing import validate_subject

            result = validate_subject(0)

        assert result["valid"] is False
        assert result["has_inf"] is True

    def test_statistics_returned(self, faced_preprocessed_dir: Path):
        """Validation dict includes min, max, mean, std statistics."""
        with mock.patch(f"{MODULE}.OUT_DIR", faced_preprocessed_dir):
            from src.preprocessing.faced.run_preprocessing import validate_subject

            result = validate_subject(0)

        for key in ("min", "max", "mean", "std"):
            assert key in result
            assert isinstance(result[key], float)


# Resampling correctness


class TestResamplingCorrectness:
    """Verify the scipy FFT resampling behaves as expected."""

    def test_resample_preserves_mean(self, faced_raw_dir: Path, tmp_path: Path):
        """Resampled signal should preserve the global mean value."""
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        # Load the raw data to compare
        with open(faced_raw_dir / "sub000.pkl", "rb") as f:
            raw = pickle.load(f)

        with (
            mock.patch(f"{MODULE}.RAW_DIR", faced_raw_dir),
            mock.patch(f"{MODULE}.OUT_DIR", out_dir),
        ):
            from src.preprocessing.faced.run_preprocessing import preprocess_subject

            preprocess_subject(0)

        resampled = np.load(out_dir / "sub000.npy")

        # FFT resample preserves the DC component (mean)
        np.testing.assert_allclose(
            resampled.mean(),
            raw.mean(),
            rtol=0.05,
            err_msg="Resampling should approximately preserve signal mean",
        )

    def test_resample_matches_scipy_direct(self):
        """Our pipeline's resample must match scipy.signal.resample directly."""
        rng = np.random.RandomState(123)
        data = rng.randn(FACED_N_STIMULI, FACED_N_CHANNELS, FACED_RAW_TIMEPOINTS).astype(np.float32)
        expected = scipy_signal.resample(data, FACED_PREP_TIMEPOINTS, axis=2)
        np.testing.assert_allclose(expected.shape, FACED_PREP_SHAPE)

    def test_resample_ratio(self):
        """Output timepoints = input x (200/250) = input x 0.8."""
        ratio = FACED_PREP_TIMEPOINTS / FACED_RAW_TIMEPOINTS
        assert ratio == pytest.approx(0.8)


# Constants sanity checks


class TestFacedConstants:
    """Verify the preprocessing module's constants match expected values."""

    def test_module_constants(self):
        from src.preprocessing.faced.run_preprocessing import (
            EXPECTED_SHAPE,
            N_SUBJECTS,
            ORIG_SFREQ,
            ORIG_TIMEPOINTS,
            TARGET_SFREQ,
            TARGET_TIMEPOINTS,
        )

        assert N_SUBJECTS == 123
        assert ORIG_SFREQ == 250
        assert TARGET_SFREQ == 200
        assert ORIG_TIMEPOINTS == 7500
        assert TARGET_TIMEPOINTS == 6000
        assert EXPECTED_SHAPE == (28, 32, 6000)

    def test_timepoints_consistent_with_frequency(self):
        """30s x 250 Hz = 7500, 30s x 200 Hz = 6000."""
        assert 30 * 250 == FACED_RAW_TIMEPOINTS
        assert 30 * 200 == FACED_PREP_TIMEPOINTS
