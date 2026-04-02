"""
Tests for the THU-EP preprocessing pipeline.

Covers:
  - Individual preprocessing steps (extract_band, remove_channels, downsample,
    z_normalize, artifact_clipping, export)
  - THUEPConfig: YAML loading, property accessors, computed properties
  - THUEPPreprocessingPipeline: end-to-end processing, validation, file discovery
  - Shape transformations through the pipeline
  - Edge cases: zero-std channels, threshold clipping, missing files
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import pytest
import yaml

from tests.conftest import (
    THUEP_N_BANDS,
    THUEP_N_CHANNELS_PREP,
    THUEP_N_CHANNELS_RAW,
    THUEP_N_STIMULI,
    THUEP_PREP_SHAPE,
    THUEP_PREP_TIMEPOINTS,
    THUEP_RAW_SHAPE,
    THUEP_RAW_TIMEPOINTS,
    _make_fake_eeg,
)

# Individual preprocessing steps


class TestExtractFrequencyBand:
    """Tests for extract_frequency_band."""

    def test_output_shape(self):
        """(7500, 32, 28, 6) → (28, 32, 7500) after extracting one band."""
        from src.preprocessing.thu_ep.preprocessing_steps import extract_frequency_band

        data = _make_fake_eeg(THUEP_RAW_SHAPE)
        result = extract_frequency_band(data, band_index=5)
        assert result.shape == (THUEP_N_STIMULI, THUEP_N_CHANNELS_RAW, THUEP_RAW_TIMEPOINTS)

    def test_correct_band_extracted(self):
        """Verify the correct frequency band index is selected."""
        from src.preprocessing.thu_ep.preprocessing_steps import extract_frequency_band

        rng = np.random.RandomState(42)
        data = rng.randn(*THUEP_RAW_SHAPE).astype(np.float32)

        for band_idx in range(THUEP_N_BANDS):
            result = extract_frequency_band(data, band_index=band_idx)
            # After transpose: result[stim, ch, t] should equal data[t, ch, stim, band_idx]
            expected_slice = data[:, :, :, band_idx].transpose(2, 1, 0)
            np.testing.assert_array_equal(result, expected_slice)

    def test_all_band_indices_valid(self):
        """All band indices 0-5 should work without error."""
        from src.preprocessing.thu_ep.preprocessing_steps import extract_frequency_band

        data = _make_fake_eeg(THUEP_RAW_SHAPE)
        for i in range(THUEP_N_BANDS):
            result = extract_frequency_band(data, band_index=i)
            assert result.ndim == 3


class TestRemoveReferenceChannels:
    """Tests for remove_reference_channels."""

    def test_output_shape(self):
        """Removing 2 channels: 32 → 30."""
        from src.preprocessing.thu_ep.preprocessing_steps import remove_reference_channels

        data = _make_fake_eeg((THUEP_N_STIMULI, THUEP_N_CHANNELS_RAW, THUEP_RAW_TIMEPOINTS))
        # A1 and A2 are at indices 16, 17 in the config channel list
        result = remove_reference_channels(data, channels_to_remove_indices=[16, 17])
        assert result.shape == (THUEP_N_STIMULI, THUEP_N_CHANNELS_PREP, THUEP_RAW_TIMEPOINTS)

    def test_correct_channels_removed(self):
        """Remaining channels must NOT include the removed ones."""
        from src.preprocessing.thu_ep.preprocessing_steps import remove_reference_channels

        n_ch = 5
        data = (
            np.arange(THUEP_N_STIMULI * n_ch * 10)
            .reshape(THUEP_N_STIMULI, n_ch, 10)
            .astype(np.float32)
        )
        result = remove_reference_channels(data, channels_to_remove_indices=[1, 3])
        assert result.shape[1] == 3  # 5 - 2 = 3
        # Channel axis should contain indices 0, 2, 4
        np.testing.assert_array_equal(result[:, 0, :], data[:, 0, :])
        np.testing.assert_array_equal(result[:, 1, :], data[:, 2, :])
        np.testing.assert_array_equal(result[:, 2, :], data[:, 4, :])

    def test_no_channels_removed(self):
        """Empty removal list leaves data unchanged."""
        from src.preprocessing.thu_ep.preprocessing_steps import remove_reference_channels

        data = _make_fake_eeg((THUEP_N_STIMULI, THUEP_N_CHANNELS_RAW, 100))
        result = remove_reference_channels(data, channels_to_remove_indices=[])
        assert result.shape == data.shape
        np.testing.assert_array_equal(result, data)


class TestDownsampleStimuli:
    """Tests for downsample_stimuli (scipy FFT resample)."""

    def test_output_shape(self):
        """7500 → 6000 samples at 250→200 Hz."""
        from src.preprocessing.thu_ep.preprocessing_steps import downsample_stimuli

        data = _make_fake_eeg((THUEP_N_STIMULI, THUEP_N_CHANNELS_PREP, THUEP_RAW_TIMEPOINTS))
        result = downsample_stimuli(data, original_sfreq=250.0, target_sfreq=200.0)
        assert result.shape == THUEP_PREP_SHAPE

    def test_preserves_mean(self):
        """Resampled signal should preserve the global mean (DC component)."""
        from src.preprocessing.thu_ep.preprocessing_steps import downsample_stimuli

        rng = np.random.RandomState(0)
        data = (
            rng.randn(THUEP_N_STIMULI, THUEP_N_CHANNELS_PREP, THUEP_RAW_TIMEPOINTS) * 50
        ).astype(np.float32)
        result = downsample_stimuli(data, original_sfreq=250.0, target_sfreq=200.0)

        np.testing.assert_allclose(
            result.mean(),
            data.mean(),
            rtol=0.05,
            err_msg="FFT resampling should preserve signal mean",
        )

    def test_identity_when_same_rate(self):
        """No change when original == target frequency (n_samples stays same)."""
        from src.preprocessing.thu_ep.preprocessing_steps import downsample_stimuli

        data = _make_fake_eeg((THUEP_N_STIMULI, THUEP_N_CHANNELS_PREP, 100))
        result = downsample_stimuli(data, original_sfreq=200.0, target_sfreq=200.0)
        assert result.shape == data.shape


class TestComputeGlobalStatistics:
    """Tests for compute_global_statistics."""

    def test_output_shapes(self):
        """Mean and std should be per-channel vectors of length n_channels."""
        from src.preprocessing.thu_ep.preprocessing_steps import compute_global_statistics

        data = _make_fake_eeg(THUEP_PREP_SHAPE)
        mean, std = compute_global_statistics(data)
        assert mean.shape == (THUEP_N_CHANNELS_PREP,)
        assert std.shape == (THUEP_N_CHANNELS_PREP,)

    def test_known_statistics(self):
        """With constant data per channel, mean should be that constant, std ≈ 0."""
        from src.preprocessing.thu_ep.preprocessing_steps import compute_global_statistics

        data = np.ones(THUEP_PREP_SHAPE, dtype=np.float32)
        data[:, 0, :] = 5.0  # channel 0 = constant 5
        data[:, 1, :] = -3.0  # channel 1 = constant -3

        mean, std = compute_global_statistics(data)
        assert mean[0] == pytest.approx(5.0)
        assert mean[1] == pytest.approx(-3.0)
        assert std[0] == pytest.approx(0.0, abs=1e-6)
        assert std[1] == pytest.approx(0.0, abs=1e-6)


class TestZNormalizeGlobal:
    """Tests for z_normalize_global."""

    def test_normalized_stats(self):
        """After z-normalization, per-channel mean ≈ 0, std ≈ 1."""
        from src.preprocessing.thu_ep.preprocessing_steps import (
            compute_global_statistics,
            z_normalize_global,
        )

        rng = np.random.RandomState(42)
        data = (rng.randn(*THUEP_PREP_SHAPE) * 100 + 50).astype(np.float32)
        mean, std = compute_global_statistics(data)
        normed = z_normalize_global(data, mean, std)

        # Check per-channel: reshape to (n_ch, -1) and verify
        for ch in range(THUEP_N_CHANNELS_PREP):
            ch_data = normed[:, ch, :].ravel()
            assert np.mean(ch_data) == pytest.approx(0.0, abs=1e-4)
            assert np.std(ch_data) == pytest.approx(1.0, abs=1e-4)

    def test_zero_std_channel_safe(self):
        """Channels with zero std should not produce NaN (safe division)."""
        from src.preprocessing.thu_ep.preprocessing_steps import z_normalize_global

        data = np.ones(THUEP_PREP_SHAPE, dtype=np.float32) * 3.0
        mean = np.full(THUEP_N_CHANNELS_PREP, 3.0)
        std = np.zeros(THUEP_N_CHANNELS_PREP)  # all-zero std

        result = z_normalize_global(data, mean, std)
        assert not np.isnan(result).any(), "NaN from zero-std division"
        # (3 - 3) / 1.0 = 0.0 (safe std replaced with 1.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)


class TestArtifactClipping:
    """Tests for artifact_clipping."""

    def test_clips_to_threshold(self):
        """Values beyond ±threshold must be clipped."""
        from src.preprocessing.thu_ep.preprocessing_steps import artifact_clipping

        data = np.array([-20.0, -5.0, 0.0, 5.0, 20.0], dtype=np.float32)
        result = artifact_clipping(data, threshold_std=10.0)
        np.testing.assert_array_equal(result, [-10.0, -5.0, 0.0, 5.0, 10.0])

    def test_no_clipping_within_range(self):
        """Data within range is untouched."""
        from src.preprocessing.thu_ep.preprocessing_steps import artifact_clipping

        data = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        result = artifact_clipping(data, threshold_std=15.0)
        np.testing.assert_array_equal(result, data)


class TestExportSubjectNpy:
    """Tests for export_subject_npy."""

    def test_exported_file_exists(self, tmp_path: Path):
        """Export creates sub_XX.npy with correct filename."""
        from src.preprocessing.thu_ep.preprocessing_steps import export_subject_npy

        data = _make_fake_eeg(THUEP_PREP_SHAPE)
        path = export_subject_npy(5, data, str(tmp_path))
        assert Path(path).exists()
        assert Path(path).name == "sub_05.npy"

    def test_exported_dtype_float32(self, tmp_path: Path):
        """Exported file must be float32."""
        from src.preprocessing.thu_ep.preprocessing_steps import export_subject_npy

        data = np.ones(THUEP_PREP_SHAPE, dtype=np.float64)  # input is float64
        path = export_subject_npy(1, data, str(tmp_path))
        loaded = np.load(path)
        assert loaded.dtype == np.float32

    def test_exported_shape_preserved(self, tmp_path: Path):
        """Exported shape must match input shape."""
        from src.preprocessing.thu_ep.preprocessing_steps import export_subject_npy

        data = _make_fake_eeg(THUEP_PREP_SHAPE)
        path = export_subject_npy(1, data, str(tmp_path))
        loaded = np.load(path)
        assert loaded.shape == THUEP_PREP_SHAPE

    def test_creates_output_directory(self, tmp_path: Path):
        """Export must create parent directories if they don't exist."""
        from src.preprocessing.thu_ep.preprocessing_steps import export_subject_npy

        nested = tmp_path / "a" / "b" / "c"
        data = _make_fake_eeg(THUEP_PREP_SHAPE)
        path = export_subject_npy(1, data, str(nested))
        assert Path(path).exists()


# THUEPConfig


class TestTHUEPConfig:
    """Tests for the THU-EP YAML config loader."""

    def test_load_from_yaml(self, tmp_path: Path, thuep_test_config: dict):
        """Config loads successfully from a YAML file."""
        config_path = tmp_path / "thu_ep.yml"
        with open(config_path, "w") as f:
            yaml.dump(thuep_test_config, f)

        from src.preprocessing.thu_ep.config import THUEPConfig

        cfg = THUEPConfig(config_path=config_path)

        assert cfg.n_subjects == 80
        assert cfg.n_stimuli == 28
        assert cfg.n_channels == 32
        assert cfg.n_bands == 6

    def test_channels_to_remove_indices(self, tmp_path: Path, thuep_test_config: dict):
        """A1 and A2 indices computed from channel list."""
        config_path = tmp_path / "thu_ep.yml"
        with open(config_path, "w") as f:
            yaml.dump(thuep_test_config, f)

        from src.preprocessing.thu_ep.config import THUEPConfig

        cfg = THUEPConfig(config_path=config_path)

        # In the config, A1 is at index 16, A2 at index 17
        indices = cfg.channels_to_remove_indices
        assert len(indices) == 2
        all_ch = cfg.all_channels
        assert all_ch[indices[0]] == "A1"
        assert all_ch[indices[1]] == "A2"

    def test_final_channels_count(self, tmp_path: Path, thuep_test_config: dict):
        """32 total - 2 removed = 30 final channels."""
        config_path = tmp_path / "thu_ep.yml"
        with open(config_path, "w") as f:
            yaml.dump(thuep_test_config, f)

        from src.preprocessing.thu_ep.config import THUEPConfig

        cfg = THUEPConfig(config_path=config_path)

        assert cfg.n_channels_final == 30
        assert len(cfg.final_channels) == 30
        assert "A1" not in cfg.final_channels
        assert "A2" not in cfg.final_channels

    def test_sampling_properties(self, tmp_path: Path, thuep_test_config: dict):
        """Sampling rate and timepoint properties."""
        config_path = tmp_path / "thu_ep.yml"
        with open(config_path, "w") as f:
            yaml.dump(thuep_test_config, f)

        from src.preprocessing.thu_ep.config import THUEPConfig

        cfg = THUEPConfig(config_path=config_path)

        assert cfg.original_sfreq == 250.0
        assert cfg.target_sfreq == 200.0
        assert cfg.original_n_samples == 7500
        assert cfg.target_n_samples == 6000
        assert cfg.downsample_factor == pytest.approx(1.25)

    def test_steps_enabled(self, tmp_path: Path, thuep_test_config: dict):
        """Step enable/disable flags read correctly."""
        config_path = tmp_path / "thu_ep.yml"
        with open(config_path, "w") as f:
            yaml.dump(thuep_test_config, f)

        from src.preprocessing.thu_ep.config import THUEPConfig

        cfg = THUEPConfig(config_path=config_path)

        assert cfg.is_step_enabled("extract_band") is True
        assert cfg.is_step_enabled("downsample") is True
        assert cfg.is_step_enabled("z_normalize") is False
        assert cfg.is_step_enabled("artifact_clipping") is False
        assert cfg.is_step_enabled("nonexistent_step") is False

    def test_missing_config_raises(self, tmp_path: Path):
        """Loading a non-existent YAML file raises FileNotFoundError."""
        from src.preprocessing.thu_ep.config import THUEPConfig

        with pytest.raises(FileNotFoundError):
            THUEPConfig(config_path=tmp_path / "does_not_exist.yml")

    def test_broad_band_index(self, tmp_path: Path, thuep_test_config: dict):
        """Broad-band index should be 5 (0.5–47 Hz)."""
        config_path = tmp_path / "thu_ep.yml"
        with open(config_path, "w") as f:
            yaml.dump(thuep_test_config, f)

        from src.preprocessing.thu_ep.config import THUEPConfig

        cfg = THUEPConfig(config_path=config_path)

        assert cfg.broad_band_index == 5
        assert cfg.band_names[5] == "broad-band"


# THUEPPreprocessingPipeline


class TestTHUEPPipeline:
    """Tests for the end-to-end THU-EP preprocessing pipeline."""

    def _make_pipeline_with_config(self, tmp_path: Path, thuep_test_config: dict):
        """Helper: write config to YAML and create a pipeline with it."""
        config_path = tmp_path / "thu_ep.yml"
        with open(config_path, "w") as f:
            yaml.dump(thuep_test_config, f)

        from src.preprocessing.thu_ep.config import THUEPConfig
        from src.preprocessing.thu_ep.thu_ep_preprocessing_pipeline import (
            THUEPPreprocessingPipeline,
        )

        cfg = THUEPConfig(config_path=config_path)
        return THUEPPreprocessingPipeline(config=cfg)

    def test_process_subject_shape(
        self, thuep_raw_dir: Path, tmp_path: Path, thuep_test_config: dict
    ):
        """End-to-end: raw .mat → preprocessed .npy with correct shape."""
        # Point config paths at the fixture directories
        thuep_test_config["paths"]["raw_data_dir"] = str(thuep_raw_dir)
        out_dir = tmp_path / "output"
        thuep_test_config["paths"]["preprocessed_output_dir"] = str(out_dir)

        pipeline = self._make_pipeline_with_config(tmp_path, thuep_test_config)
        files = pipeline.get_subject_files()
        assert len(files) == 3  # sub_1, sub_2, sub_3

        result = pipeline.process_subject(files[0])
        assert result["success"] is True

        # Load output and check shape
        arr = np.load(result["output_file"])
        assert arr.shape == THUEP_PREP_SHAPE

    def test_process_subject_no_nan(
        self, thuep_raw_dir: Path, tmp_path: Path, thuep_test_config: dict
    ):
        """Processing must not introduce NaN or Inf."""
        thuep_test_config["paths"]["raw_data_dir"] = str(thuep_raw_dir)
        out_dir = tmp_path / "output"
        thuep_test_config["paths"]["preprocessed_output_dir"] = str(out_dir)

        pipeline = self._make_pipeline_with_config(tmp_path, thuep_test_config)
        files = pipeline.get_subject_files()
        result = pipeline.process_subject(files[0])

        arr = np.load(result["output_file"])
        assert not np.isnan(arr).any()
        assert not np.isinf(arr).any()

    def test_process_all_subjects(
        self, thuep_raw_dir: Path, tmp_path: Path, thuep_test_config: dict
    ):
        """process_all_subjects processes all found .mat files."""
        thuep_test_config["paths"]["raw_data_dir"] = str(thuep_raw_dir)
        out_dir = tmp_path / "output"
        thuep_test_config["paths"]["preprocessed_output_dir"] = str(out_dir)

        pipeline = self._make_pipeline_with_config(tmp_path, thuep_test_config)
        results = pipeline.process_all_subjects()

        assert results["total_subjects"] == 3
        assert results["successful"] == 3
        assert results["failed"] == 0

    def test_get_subject_id_from_path(self, tmp_path: Path, thuep_test_config: dict):
        """Pipeline extracts subject ID correctly from filename."""
        pipeline = self._make_pipeline_with_config(tmp_path, thuep_test_config)
        assert pipeline.get_subject_id(Path("sub_42.mat")) == 42
        assert pipeline.get_subject_id(Path("/some/path/sub_7.mat")) == 7

    def test_validate_preprocessed_data(
        self, thuep_preprocessed_dir: Path, tmp_path: Path, thuep_test_config: dict
    ):
        """Validation passes for a well-formed preprocessed file."""
        thuep_test_config["paths"]["preprocessed_output_dir"] = str(thuep_preprocessed_dir)

        pipeline = self._make_pipeline_with_config(tmp_path, thuep_test_config)
        result = pipeline.validate_preprocessed_data(1)

        assert result["valid"] is True
        assert result["shape"] == THUEP_PREP_SHAPE
        assert result["dtype_valid"] is True

    def test_load_preprocessed_missing_raises(self, tmp_path: Path, thuep_test_config: dict):
        """Loading a non-existent preprocessed file raises FileNotFoundError."""
        thuep_test_config["paths"]["preprocessed_output_dir"] = str(tmp_path / "empty")
        pipeline = self._make_pipeline_with_config(tmp_path, thuep_test_config)

        with pytest.raises(FileNotFoundError):
            pipeline.load_preprocessed_subject(99)

    def test_get_subject_files_empty_dir(self, tmp_path: Path, thuep_test_config: dict):
        """get_subject_files raises FileNotFoundError for missing raw dir."""
        thuep_test_config["paths"]["raw_data_dir"] = str(tmp_path / "nonexistent")
        pipeline = self._make_pipeline_with_config(tmp_path, thuep_test_config)

        with pytest.raises(FileNotFoundError):
            pipeline.get_subject_files()


# Shape transformation integration test


class TestShapeTransformations:
    """Verify the full chain of shape transformations matches documentation."""

    def test_full_pipeline_shapes(self):
        """
        (7500, 32, 28, 6) → extract_band → (28, 32, 7500)
        → remove_channels → (28, 30, 7500)
        → downsample → (28, 30, 6000)
        """
        from src.preprocessing.thu_ep.preprocessing_steps import (
            downsample_stimuli,
            extract_frequency_band,
            remove_reference_channels,
        )

        data = _make_fake_eeg(THUEP_RAW_SHAPE)

        # Step 1: extract band
        step1 = extract_frequency_band(data, band_index=5)
        assert step1.shape == (28, 32, 7500)

        # Step 2: remove A1, A2 (indices 16, 17 in config)
        step2 = remove_reference_channels(step1, channels_to_remove_indices=[16, 17])
        assert step2.shape == (28, 30, 7500)

        # Step 3: downsample
        step3 = downsample_stimuli(step2, original_sfreq=250.0, target_sfreq=200.0)
        assert step3.shape == (28, 30, 6000)
