"""
Tests for the dataset classes (base, FACED, THU-EP).

Covers:
  - Label mapping: 9-class and binary modes, neutral dropping in binary
  - EEGWindowDataset: sliding window index, __getitem__, scale_factor, labels
  - FACEDWindowDataset: file naming, channel count, no exclusions
  - THUEPWindowDataset: file naming, channel count, subject/stimuli exclusions
  - Stimulus filtering
  - Edge cases: overlapping windows, single subject, excluded-only subjects
  - build_raw_dataset factory
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.conftest import (
    FACED_N_CHANNELS,
    FACED_N_STIMULI,
    FACED_PREP_SHAPE,
    FACED_PREP_TIMEPOINTS,
    THUEP_N_CHANNELS_PREP,
    THUEP_N_STIMULI,
    THUEP_PREP_SHAPE,
    THUEP_PREP_TIMEPOINTS,
    _make_fake_eeg,
)

# Label mapping


class TestBuildStimulusLabelMap:
    """Tests for build_stimulus_label_map."""

    def test_9class_all_mapped(self):
        """All 28 stimuli have integer labels 0-8 in 9-class mode."""
        from src.datasets.base import build_stimulus_label_map

        lmap = build_stimulus_label_map("9-class")
        assert len(lmap) == 28
        assert all(isinstance(v, int) for v in lmap.values())
        assert set(lmap.values()) == set(range(9))

    def test_9class_label_distribution(self):
        """Anger/Disgust/Fear/Sadness have 3 stimuli each, Neutral has 4, rest have 3."""
        from src.datasets.base import build_stimulus_label_map

        lmap = build_stimulus_label_map("9-class")
        from collections import Counter

        counts = Counter(lmap.values())
        # classes 0-3, 5-8 have 3 stimuli; class 4 (neutral) has 4
        for cls in [0, 1, 2, 3, 5, 6, 7, 8]:
            assert counts[cls] == 3
        assert counts[4] == 4

    def test_binary_neutral_dropped(self):
        """In binary mode, stimuli 12-15 (Neutral) map to None."""
        from src.datasets.base import build_stimulus_label_map

        lmap = build_stimulus_label_map("binary")
        for stim in [12, 13, 14, 15]:
            assert lmap[stim] is None

    def test_binary_neg_pos(self):
        """In binary mode: anger/disgust/fear/sadness → 0, amusement/inspiration/joy/tenderness → 1."""
        from src.datasets.base import build_stimulus_label_map

        lmap = build_stimulus_label_map("binary")
        for stim in range(0, 12):  # classes 0-3 → negative
            assert lmap[stim] == 0
        for stim in range(16, 28):  # classes 5-8 → positive
            assert lmap[stim] == 1

    def test_invalid_task_mode(self):
        """Invalid task_mode raises AssertionError."""
        from src.datasets.base import build_stimulus_label_map

        with pytest.raises(AssertionError):
            build_stimulus_label_map("3-class")


class TestStimulusLabelsArray:
    """Tests for the STIMULUS_LABELS constant."""

    def test_length(self):
        from src.datasets.base import STIMULUS_LABELS

        assert len(STIMULUS_LABELS) == 28

    def test_values_range(self):
        from src.datasets.base import STIMULUS_LABELS

        assert STIMULUS_LABELS.min() == 0
        assert STIMULUS_LABELS.max() == 8


# FACEDWindowDataset


class TestFACEDWindowDataset:
    """Tests for the FACED dataset class."""

    def test_length_9class_nonoverlapping(self, faced_preprocessed_dir: Path):
        """Dataset length = n_subjects x 28_stimuli x n_windows."""
        from src.datasets.faced_dataset import FACEDWindowDataset

        ds = FACEDWindowDataset(
            subject_ids=[0, 1, 2],
            task_mode="9-class",
            data_root=faced_preprocessed_dir,
            window_size=2000,
            stride=2000,
        )
        n_windows = (FACED_PREP_TIMEPOINTS - 2000) // 2000 + 1  # = 3
        expected = 3 * FACED_N_STIMULI * n_windows
        assert len(ds) == expected

    def test_length_binary_fewer_stimuli(self, faced_preprocessed_dir: Path):
        """Binary mode drops neutral stimuli (12-15), so fewer samples."""
        from src.datasets.faced_dataset import FACEDWindowDataset

        ds_9 = FACEDWindowDataset(
            subject_ids=[0],
            task_mode="9-class",
            data_root=faced_preprocessed_dir,
        )
        ds_bin = FACEDWindowDataset(
            subject_ids=[0],
            task_mode="binary",
            data_root=faced_preprocessed_dir,
        )
        # Binary has 24 stimuli (28 - 4 neutral), 9-class has 28
        assert len(ds_bin) < len(ds_9)
        ratio = len(ds_bin) / len(ds_9)
        assert ratio == pytest.approx(24 / 28)

    def test_getitem_shape(self, faced_preprocessed_dir: Path):
        """__getitem__ returns (tensor [C, window_size], int label)."""
        from src.datasets.faced_dataset import FACEDWindowDataset

        ds = FACEDWindowDataset(
            subject_ids=[0],
            task_mode="9-class",
            data_root=faced_preprocessed_dir,
            window_size=2000,
        )
        tensor, label = ds[0]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (FACED_N_CHANNELS, 2000)
        assert tensor.dtype == torch.float32
        assert isinstance(label, int)
        assert 0 <= label <= 8

    def test_scale_factor_applied(self, faced_preprocessed_dir: Path):
        """Returned tensor values should be raw / scale_factor."""
        from src.datasets.faced_dataset import FACEDWindowDataset

        ds = FACEDWindowDataset(
            subject_ids=[0],
            task_mode="9-class",
            data_root=faced_preprocessed_dir,
            window_size=2000,
            scale_factor=1000.0,
        )
        tensor, _ = ds[0]

        # Load raw and compare
        raw = np.load(faced_preprocessed_dir / "sub000.npy")
        sid, stim, start = ds.index[0]
        expected = raw[stim, :, start : start + 2000].astype(np.float32) / 1000.0
        np.testing.assert_allclose(tensor.numpy(), expected, atol=1e-6)

    def test_labels_property(self, faced_preprocessed_dir: Path):
        """labels property returns correct labels for all windows."""
        from src.datasets.faced_dataset import FACEDWindowDataset

        ds = FACEDWindowDataset(
            subject_ids=[0],
            task_mode="9-class",
            data_root=faced_preprocessed_dir,
        )
        labels = ds.labels
        assert len(labels) == len(ds)
        assert all(isinstance(l, int) for l in labels)
        assert all(0 <= l <= 8 for l in labels)

    def test_overlapping_windows(self, faced_preprocessed_dir: Path):
        """Overlapping stride produces more windows than non-overlapping."""
        from src.datasets.faced_dataset import FACEDWindowDataset

        ds_nonoverlap = FACEDWindowDataset(
            subject_ids=[0],
            task_mode="9-class",
            data_root=faced_preprocessed_dir,
            window_size=2000,
            stride=2000,
        )
        ds_overlap = FACEDWindowDataset(
            subject_ids=[0],
            task_mode="9-class",
            data_root=faced_preprocessed_dir,
            window_size=2000,
            stride=1000,
        )
        assert len(ds_overlap) > len(ds_nonoverlap)

    def test_missing_file_raises(self, tmp_path: Path):
        """Requesting a non-existent subject file raises FileNotFoundError."""
        from src.datasets.faced_dataset import FACEDWindowDataset

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            FACEDWindowDataset(
                subject_ids=[999],
                task_mode="9-class",
                data_root=empty_dir,
            )

    def test_subject_path_format(self, faced_preprocessed_dir: Path):
        """FACED uses sub{NNN}.npy naming (3-digit zero-padded)."""
        from src.datasets.faced_dataset import FACEDWindowDataset

        ds = FACEDWindowDataset(
            subject_ids=[0],
            task_mode="9-class",
            data_root=faced_preprocessed_dir,
        )
        path = ds._subject_path(0)
        assert path.name == "sub000.npy"
        path = ds._subject_path(122)
        assert path.name == "sub122.npy"

    def test_no_excluded_subjects(self):
        """FACED has no excluded subjects."""
        from src.datasets.faced_dataset import EXCLUDED_SUBJECTS

        assert EXCLUDED_SUBJECTS == set()

    def test_no_excluded_stimuli(self):
        """FACED has no excluded stimuli."""
        from src.datasets.faced_dataset import EXCLUDED_STIMULI

        assert EXCLUDED_STIMULI == {}

    def test_channel_count(self):
        """FACED has 32 channels."""
        from src.datasets.faced_dataset import FACED_CHANNELS

        assert len(FACED_CHANNELS) == 32


# THUEPWindowDataset


class TestTHUEPWindowDataset:
    """Tests for the THU-EP dataset class."""

    def test_length_9class(self, thuep_preprocessed_dir: Path):
        """Dataset length for THU-EP with 5 subjects (none excluded in fixture)."""
        from src.datasets.thu_ep_dataset import THUEPWindowDataset

        ds = THUEPWindowDataset(
            subject_ids=[1, 2, 3, 4, 5],
            task_mode="9-class",
            data_root=thuep_preprocessed_dir,
            window_size=2000,
            stride=2000,
        )
        n_windows = (THUEP_PREP_TIMEPOINTS - 2000) // 2000 + 1  # = 3
        expected = 5 * THUEP_N_STIMULI * n_windows
        assert len(ds) == expected

    def test_getitem_shape(self, thuep_preprocessed_dir: Path):
        """__getitem__ returns (tensor [30, window_size], int label)."""
        from src.datasets.thu_ep_dataset import THUEPWindowDataset

        ds = THUEPWindowDataset(
            subject_ids=[1],
            task_mode="9-class",
            data_root=thuep_preprocessed_dir,
            window_size=2000,
        )
        tensor, label = ds[0]
        assert tensor.shape == (THUEP_N_CHANNELS_PREP, 2000)
        assert tensor.dtype == torch.float32

    def test_excluded_subject_75_skipped(self, thuep_preprocessed_dir: Path):
        """Subject 75 is excluded — if requested, it's silently skipped."""
        from src.datasets.thu_ep_dataset import THUEPWindowDataset

        # Create a fake file for subject 75 so it doesn't raise FileNotFoundError
        np.save(
            thuep_preprocessed_dir / "sub_75.npy",
            _make_fake_eeg(THUEP_PREP_SHAPE),
        )

        ds_with_75 = THUEPWindowDataset(
            subject_ids=[1, 75],
            task_mode="9-class",
            data_root=thuep_preprocessed_dir,
            window_size=2000,
            stride=2000,
        )
        ds_without = THUEPWindowDataset(
            subject_ids=[1],
            task_mode="9-class",
            data_root=thuep_preprocessed_dir,
            window_size=2000,
            stride=2000,
        )
        # Subject 75 excluded → both datasets should have the same length
        assert len(ds_with_75) == len(ds_without)

    def test_excluded_stimuli_subject_37(self, thuep_preprocessed_dir: Path):
        """Subject 37's corrupted stimuli (15, 21, 24) are excluded."""
        from src.datasets.thu_ep_dataset import THUEPWindowDataset

        np.save(
            thuep_preprocessed_dir / "sub_37.npy",
            _make_fake_eeg(THUEP_PREP_SHAPE),
        )

        ds = THUEPWindowDataset(
            subject_ids=[37],
            task_mode="9-class",
            data_root=thuep_preprocessed_dir,
            window_size=2000,
            stride=2000,
        )
        n_windows = (THUEP_PREP_TIMEPOINTS - 2000) // 2000 + 1
        # 28 stimuli - 3 excluded = 25 valid stimuli
        expected = 25 * n_windows
        assert len(ds) == expected

    def test_excluded_stimuli_subject_46(self, thuep_preprocessed_dir: Path):
        """Subject 46's corrupted stimuli (3, 9, 17, 23, 26) are excluded."""
        from src.datasets.thu_ep_dataset import THUEPWindowDataset

        np.save(
            thuep_preprocessed_dir / "sub_46.npy",
            _make_fake_eeg(THUEP_PREP_SHAPE),
        )

        ds = THUEPWindowDataset(
            subject_ids=[46],
            task_mode="9-class",
            data_root=thuep_preprocessed_dir,
            window_size=2000,
            stride=2000,
        )
        n_windows = (THUEP_PREP_TIMEPOINTS - 2000) // 2000 + 1
        # 28 stimuli - 5 excluded = 23 valid stimuli
        expected = 23 * n_windows
        assert len(ds) == expected

    def test_subject_path_format(self, thuep_preprocessed_dir: Path):
        """THU-EP uses sub_{XX}.npy naming (2-digit zero-padded)."""
        from src.datasets.thu_ep_dataset import THUEPWindowDataset

        ds = THUEPWindowDataset(
            subject_ids=[1],
            task_mode="9-class",
            data_root=thuep_preprocessed_dir,
        )
        assert ds._subject_path(1).name == "sub_01.npy"
        assert ds._subject_path(80).name == "sub_80.npy"

    def test_excluded_subjects_constant(self):
        """THU-EP excludes subject 75."""
        from src.datasets.thu_ep_dataset import EXCLUDED_SUBJECTS

        assert EXCLUDED_SUBJECTS == {75}

    def test_excluded_stimuli_constants(self):
        """THU-EP has corrupted stimuli for subjects 37 and 46."""
        from src.datasets.thu_ep_dataset import EXCLUDED_STIMULI

        assert 37 in EXCLUDED_STIMULI
        assert EXCLUDED_STIMULI[37] == {15, 21, 24}
        assert 46 in EXCLUDED_STIMULI
        assert EXCLUDED_STIMULI[46] == {3, 9, 17, 23, 26}

    def test_channel_count(self):
        """THU-EP has 30 channels."""
        from src.datasets.thu_ep_dataset import THU_EP_CHANNELS

        assert len(THU_EP_CHANNELS) == 30


# Stimulus filtering


class TestStimulusFiltering:
    """Tests for the stimulus_filter parameter."""

    def test_filter_reduces_samples(self, faced_preprocessed_dir: Path):
        """Providing a stimulus filter should produce fewer samples."""
        from src.datasets.faced_dataset import FACEDWindowDataset

        ds_all = FACEDWindowDataset(
            subject_ids=[0],
            task_mode="9-class",
            data_root=faced_preprocessed_dir,
        )
        ds_filtered = FACEDWindowDataset(
            subject_ids=[0],
            task_mode="9-class",
            data_root=faced_preprocessed_dir,
            stimulus_filter={0, 1, 2},  # only Anger stimuli
        )
        assert len(ds_filtered) < len(ds_all)

    def test_filter_correct_labels(self, faced_preprocessed_dir: Path):
        """Filtered dataset should only contain labels for the filtered stimuli."""
        from src.datasets.faced_dataset import FACEDWindowDataset

        ds = FACEDWindowDataset(
            subject_ids=[0],
            task_mode="9-class",
            data_root=faced_preprocessed_dir,
            stimulus_filter={0, 1, 2},  # Anger → label 0
        )
        assert all(l == 0 for l in ds.labels)

    def test_empty_filter_empty_dataset(self, faced_preprocessed_dir: Path):
        """An empty stimulus filter produces zero samples."""
        from src.datasets.faced_dataset import FACEDWindowDataset

        ds = FACEDWindowDataset(
            subject_ids=[0],
            task_mode="9-class",
            data_root=faced_preprocessed_dir,
            stimulus_filter=set(),
        )
        assert len(ds) == 0


# build_raw_dataset factory


class TestBuildRawDataset:
    """Tests for the shared dataset factory."""

    def test_returns_faced(self, faced_preprocessed_dir: Path):
        """Factory returns FACEDWindowDataset when dataset='faced'."""
        from src.approaches.shared.dataset import build_raw_dataset
        from src.datasets.faced_dataset import FACEDWindowDataset

        cfg = SimpleNamespace(
            dataset="faced",
            task_mode="9-class",
            data_root=faced_preprocessed_dir,
            window_size=2000,
            stride=2000,
            scale_factor=1000.0,
        )
        ds = build_raw_dataset(cfg, subject_ids=[0])
        assert isinstance(ds, FACEDWindowDataset)

    def test_returns_thuep(self, thuep_preprocessed_dir: Path):
        """Factory returns THUEPWindowDataset when dataset='thu-ep'."""
        from src.approaches.shared.dataset import build_raw_dataset
        from src.datasets.thu_ep_dataset import THUEPWindowDataset

        cfg = SimpleNamespace(
            dataset="thu-ep",
            task_mode="9-class",
            data_root=thuep_preprocessed_dir,
            window_size=2000,
            stride=2000,
            scale_factor=1000.0,
        )
        ds = build_raw_dataset(cfg, subject_ids=[1])
        assert isinstance(ds, THUEPWindowDataset)

    def test_factory_passes_stimulus_filter(self, faced_preprocessed_dir: Path):
        """Factory forwards stimulus_filter to the dataset class."""
        from src.approaches.shared.dataset import build_raw_dataset

        cfg = SimpleNamespace(
            dataset="faced",
            task_mode="9-class",
            data_root=faced_preprocessed_dir,
            window_size=2000,
            stride=2000,
            scale_factor=1000.0,
        )
        ds = build_raw_dataset(cfg, subject_ids=[0], stimulus_filter={0, 1, 2})
        assert all(l == 0 for l in ds.labels)
