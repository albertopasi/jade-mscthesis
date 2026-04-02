"""
Tests for fold/split utilities (folds.py).

Covers:
  - get_all_subjects: correct IDs for FACED and THU-EP, exclusion handling
  - get_kfold_splits: fold count, disjoint train/val, reproducibility
  - get_stimulus_generalization_split: balanced 2/3-1/3 split, neutral handling
  - get_official_split: REVE static split for FACED
  - Edge cases: unknown dataset, binary mode neutral exclusion
"""

from __future__ import annotations

import numpy as np
import pytest

# get_all_subjects


class TestGetAllSubjects:
    """Tests for get_all_subjects."""

    def test_faced_count(self):
        """FACED has 123 subjects (0-122), none excluded."""
        from src.datasets.folds import get_all_subjects

        subjects = get_all_subjects("faced")
        assert len(subjects) == 123
        assert subjects[0] == 0
        assert subjects[-1] == 122

    def test_faced_0_indexed(self):
        """FACED subject IDs are 0-indexed (0–122)."""
        from src.datasets.folds import get_all_subjects

        subjects = get_all_subjects("faced")
        assert subjects == list(range(123))

    def test_thuep_count(self):
        """THU-EP has 79 usable subjects (1-80, excluding 75)."""
        from src.datasets.folds import get_all_subjects

        subjects = get_all_subjects("thu-ep")
        assert len(subjects) == 79

    def test_thuep_1_indexed(self):
        """THU-EP subject IDs are 1-indexed (1–80)."""
        from src.datasets.folds import get_all_subjects

        subjects = get_all_subjects("thu-ep")
        assert subjects[0] == 1
        assert subjects[-1] == 80

    def test_thuep_excludes_75(self):
        """THU-EP must not include subject 75 (corrupted)."""
        from src.datasets.folds import get_all_subjects

        subjects = get_all_subjects("thu-ep")
        assert 75 not in subjects

    def test_unknown_dataset_raises(self):
        """Unknown dataset name raises ValueError."""
        from src.datasets.folds import get_all_subjects

        with pytest.raises(ValueError, match="Unknown dataset"):
            get_all_subjects("imagenet")

    def test_faced_subjects_sorted(self):
        """Returned subject list should be sorted."""
        from src.datasets.folds import get_all_subjects

        subjects = get_all_subjects("faced")
        assert subjects == sorted(subjects)

    def test_thuep_subjects_sorted(self):
        """Returned subject list should be sorted."""
        from src.datasets.folds import get_all_subjects

        subjects = get_all_subjects("thu-ep")
        assert subjects == sorted(subjects)


# get_kfold_splits


class TestGetKFoldSplits:
    """Tests for get_kfold_splits."""

    def test_default_10_folds(self):
        """Default produces 10 folds."""
        from src.datasets.folds import get_kfold_splits

        subjects = list(range(100))
        splits = get_kfold_splits(subjects)
        assert len(splits) == 10

    def test_custom_fold_count(self):
        """Custom n_folds parameter works."""
        from src.datasets.folds import get_kfold_splits

        subjects = list(range(50))
        splits = get_kfold_splits(subjects, n_folds=5)
        assert len(splits) == 5

    def test_train_val_disjoint(self):
        """Train and val indices must be disjoint within each fold."""
        from src.datasets.folds import get_kfold_splits

        subjects = list(range(100))
        splits = get_kfold_splits(subjects)

        for train_idx, val_idx in splits:
            train_set = set(train_idx)
            val_set = set(val_idx)
            assert train_set.isdisjoint(val_set), "Train/val overlap detected"

    def test_all_subjects_covered(self):
        """Union of train+val across a single fold covers all subjects."""
        from src.datasets.folds import get_kfold_splits

        subjects = list(range(100))
        splits = get_kfold_splits(subjects)

        for train_idx, val_idx in splits:
            all_idx = set(train_idx) | set(val_idx)
            assert all_idx == set(range(100))

    def test_each_subject_in_val_once(self):
        """Each subject appears in exactly one validation fold."""
        from src.datasets.folds import get_kfold_splits

        subjects = list(range(100))
        splits = get_kfold_splits(subjects)

        val_counts = np.zeros(100, dtype=int)
        for _, val_idx in splits:
            for i in val_idx:
                val_counts[i] += 1
        assert (val_counts == 1).all(), "Some subjects appear in val more/less than once"

    def test_reproducibility(self):
        """Same seed produces identical splits."""
        from src.datasets.folds import get_kfold_splits

        subjects = list(range(79))
        splits_a = get_kfold_splits(subjects, random_state=42)
        splits_b = get_kfold_splits(subjects, random_state=42)

        for (train_a, val_a), (train_b, val_b) in zip(splits_a, splits_b):
            np.testing.assert_array_equal(train_a, train_b)
            np.testing.assert_array_equal(val_a, val_b)

    def test_different_seeds_differ(self):
        """Different seeds produce different splits."""
        from src.datasets.folds import get_kfold_splits

        subjects = list(range(100))
        splits_a = get_kfold_splits(subjects, random_state=42)
        splits_b = get_kfold_splits(subjects, random_state=99)

        # At least one fold should differ
        any_different = False
        for (_, val_a), (_, val_b) in zip(splits_a, splits_b):
            if not np.array_equal(val_a, val_b):
                any_different = True
                break
        assert any_different


# get_stimulus_generalization_split


class TestGetStimulusGeneralizationSplit:
    """Tests for get_stimulus_generalization_split."""

    def test_9class_train_test_disjoint(self):
        """Train and test stimulus sets must be disjoint."""
        from src.datasets.folds import get_stimulus_generalization_split

        train, test = get_stimulus_generalization_split("9-class")
        assert train.isdisjoint(test)

    def test_9class_covers_all_28(self):
        """Train + test should cover all 28 stimuli in 9-class mode."""
        from src.datasets.folds import get_stimulus_generalization_split

        train, test = get_stimulus_generalization_split("9-class")
        assert train | test == set(range(28))

    def test_9class_approximate_split_ratio(self):
        """Train ≈ 2/3 of stimuli, test ≈ 1/3."""
        from src.datasets.folds import get_stimulus_generalization_split

        train, test = get_stimulus_generalization_split("9-class")
        # 9 groups: 8 groups of 3 (2 train, 1 test) + 1 group of 4 (3 train, 1 test)
        # = 8×2 + 3 = 19 train, 8×1 + 1 = 9 test
        assert len(train) == 19
        assert len(test) == 9

    def test_9class_balanced_per_emotion(self):
        """Each emotion group contributes to both train and test sets."""
        from src.datasets.base import STIMULUS_LABELS
        from src.datasets.folds import get_stimulus_generalization_split

        train, test = get_stimulus_generalization_split("9-class")

        # Verify each of the 9 classes has at least 1 train and 1 test stimulus
        for cls in range(9):
            class_stimuli = {i for i, l in enumerate(STIMULUS_LABELS) if l == cls}
            assert class_stimuli & train, f"Class {cls} has no train stimuli"
            assert class_stimuli & test, f"Class {cls} has no test stimuli"

    def test_binary_excludes_neutral(self):
        """In binary mode, neutral stimuli (12-15) appear in neither set."""
        from src.datasets.folds import get_stimulus_generalization_split

        train, test = get_stimulus_generalization_split("binary")
        neutral = {12, 13, 14, 15}
        assert train.isdisjoint(neutral)
        assert test.isdisjoint(neutral)

    def test_binary_fewer_stimuli(self):
        """Binary mode has fewer total stimuli than 9-class (neutral dropped)."""
        from src.datasets.folds import get_stimulus_generalization_split

        train_9, test_9 = get_stimulus_generalization_split("9-class")
        train_b, test_b = get_stimulus_generalization_split("binary")
        assert len(train_b) + len(test_b) < len(train_9) + len(test_9)

    def test_reproducibility(self):
        """Same seed produces identical splits."""
        from src.datasets.folds import get_stimulus_generalization_split

        train_a, test_a = get_stimulus_generalization_split("9-class", seed=123)
        train_b, test_b = get_stimulus_generalization_split("9-class", seed=123)
        assert train_a == train_b
        assert test_a == test_b

    def test_different_seeds_may_differ(self):
        """Different seeds may produce different splits."""
        from src.datasets.folds import get_stimulus_generalization_split

        train_a, test_a = get_stimulus_generalization_split("9-class", seed=123)
        train_b, test_b = get_stimulus_generalization_split("9-class", seed=999)
        # With different seeds, at least train or test should differ
        # (very unlikely to be identical for all 9 groups)
        assert train_a != train_b or test_a != test_b


# get_official_split


class TestGetOfficialSplit:
    """Tests for get_official_split (REVE static split)."""

    def test_faced_split_ranges(self):
        """FACED official split: train 0-79, val 80-99, test 100-122."""
        from src.datasets.folds import get_official_split

        train, val, test = get_official_split("faced")
        assert train == list(range(0, 80))
        assert val == list(range(80, 100))
        assert test == list(range(100, 123))

    def test_faced_split_sizes(self):
        """Split sizes: 80 train, 20 val, 23 test = 123 total."""
        from src.datasets.folds import get_official_split

        train, val, test = get_official_split("faced")
        assert len(train) == 80
        assert len(val) == 20
        assert len(test) == 23
        assert len(train) + len(val) + len(test) == 123

    def test_faced_split_disjoint(self):
        """Train, val, test are pairwise disjoint."""
        from src.datasets.folds import get_official_split

        train, val, test = get_official_split("faced")
        s_train, s_val, s_test = set(train), set(val), set(test)
        assert s_train.isdisjoint(s_val)
        assert s_train.isdisjoint(s_test)
        assert s_val.isdisjoint(s_test)

    def test_non_faced_raises(self):
        """Official split is only defined for FACED."""
        from src.datasets.folds import get_official_split

        with pytest.raises(ValueError, match="only defined for FACED"):
            get_official_split("thu-ep")


# Constants


class TestFoldsConstants:
    """Verify module-level constants match expected values."""

    def test_constants(self):
        from src.datasets.folds import FOLD_RANDOM_STATE, N_FOLDS, STIMULUS_SPLIT_SEED

        assert N_FOLDS == 10
        assert FOLD_RANDOM_STATE == 42
        assert STIMULUS_SPLIT_SEED == 123
