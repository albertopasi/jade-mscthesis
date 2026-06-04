"""Tests for the inference pipeline plumbing.

Covers:
  - run_fold_inference: the per-window subject/stimulus ID recovery contract,
    the num_workers=0 shuffle=False order invariant, per-subject accuracy
    computation, support for missing subjects.
  - write_run_summary: the JSON+NPZ schema that downstream tools (statistical
    tests, visualization scripts) depend on.

A stub model + stub dataset are used so we don't need REVE weights, GPU, or
real EEG data. The tests focus on the contract, not the model's accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.inference.aggregate import write_run_summary
from src.inference.predictions import FoldPredictions, run_fold_inference

# ── Test doubles ──────────────────────────────────────────────────────────


class _StubDataset:
    """Minimal dataset that mimics EEGWindowDataset's contract.

    Yields (feature, label) and exposes `.index` as list of
    (subject_id, stimulus_id, window_start). Features are deterministic so
    the stub model below produces predictable logits.
    """

    def __init__(self, index: list[tuple[int, int, int]], n_classes: int = 3, dim: int = 4):
        self.index = index
        self.n_classes = n_classes
        self.dim = dim
        # Build a deterministic feature for each window keyed off (sid, stim, start).
        rng = np.random.default_rng(42)
        self._features = rng.normal(size=(len(index), dim)).astype(np.float32)
        # Labels: stimulus_id % n_classes so test windows have non-trivial labels.
        self._labels = np.array([row[1] % n_classes for row in index], dtype=np.int64)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        return torch.from_numpy(self._features[i]), int(self._labels[i])


class _PerfectModel(nn.Module):
    """Model that maps each input directly to a one-hot logit matching the label.

    Used together with the stub dataset to make per-subject accuracy = 1.0.
    """

    def __init__(self, dataset: _StubDataset):
        super().__init__()
        self.dataset = dataset

    def forward(self, x):
        # x has shape (B, dim) but we don't actually need x —
        # we look up the label per row in the stub dataset's labels array.
        # In a real pipeline x would be EEG; here we cheat to get deterministic accuracy.
        # We use the feature itself as a hash key to retrieve the index in the dataset.
        B = x.shape[0]
        # Match each input row to the dataset's feature table by exact equality.
        # (The dataset is small enough that this is fine.)
        labels = []
        for b in range(B):
            target = x[b].cpu().numpy()
            matches = np.where(np.all(np.isclose(self.dataset._features, target), axis=1))[0]
            assert len(matches) >= 1
            labels.append(int(self.dataset._labels[matches[0]]))
        out = torch.full((B, self.dataset.n_classes), -10.0)
        for b, lbl in enumerate(labels):
            out[b, lbl] = 10.0
        return out


class _ConstantPredictModel(nn.Module):
    """Always predicts class 0 regardless of input. Used to test mixed accuracy."""

    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, x):
        B = x.shape[0]
        out = torch.full((B, self.n_classes), -10.0)
        out[:, 0] = 10.0
        return out


# ── run_fold_inference ────────────────────────────────────────────────────


class TestRunFoldInference:
    def test_raises_when_dataset_has_no_index(self):
        class _NoIndex:
            def __len__(self):
                return 0

            def __getitem__(self, i):
                raise IndexError

        with pytest.raises(TypeError, match="must expose a `.index`"):
            run_fold_inference(
                _PerfectModel(_StubDataset([(0, 0, 0)])),
                _NoIndex(),
                val_subject_ids=[0],
                batch_size=2,
                device="cpu",
                use_amp=False,
            )

    def test_basic_shape_perfect_model(self):
        # 3 subjects × 4 windows each = 12 rows.
        index = []
        for sid in range(3):
            for stim in range(4):
                index.append((sid, stim, stim * 200))
        ds = _StubDataset(index, n_classes=3)
        model = _PerfectModel(ds)

        fp = run_fold_inference(
            model, ds, val_subject_ids=[0, 1, 2], batch_size=4, device="cpu", use_amp=False
        )

        assert isinstance(fp, FoldPredictions)
        assert fp.y_true.shape == (12,)
        assert fp.y_pred.shape == (12,)
        assert fp.y_prob.shape == (12, 3)
        assert fp.subj_ids.shape == (12,)
        assert fp.stim_ids.shape == (12,)
        assert fp.window_starts.shape == (12,)

    def test_subject_id_recovery_matches_index(self):
        index = [(0, 0, 0), (0, 1, 200), (1, 0, 0), (1, 1, 200), (2, 0, 0)]
        ds = _StubDataset(index, n_classes=2)
        model = _PerfectModel(ds)

        fp = run_fold_inference(
            model, ds, val_subject_ids=[0, 1, 2], batch_size=2, device="cpu", use_amp=False
        )

        np.testing.assert_array_equal(fp.subj_ids, [0, 0, 1, 1, 2])
        np.testing.assert_array_equal(fp.stim_ids, [0, 1, 0, 1, 0])
        np.testing.assert_array_equal(fp.window_starts, [0, 200, 0, 200, 0])

    def test_perfect_model_gives_perfect_accuracy(self):
        index = [(sid, stim, 0) for sid in range(2) for stim in range(3)]
        ds = _StubDataset(index, n_classes=3)
        model = _PerfectModel(ds)

        fp = run_fold_inference(
            model, ds, val_subject_ids=[0, 1], batch_size=3, device="cpu", use_amp=False
        )

        assert fp.window_acc == 1.0
        assert fp.per_subject_acc == {0: 1.0, 1: 1.0}
        assert fp.per_subject_support == {0: 3, 1: 3}

    def test_constant_model_gives_partial_accuracy(self):
        # Stim_id % n_classes is the label → class 0 only for stim 0, 3, 6, ...
        index = [(0, stim, 0) for stim in range(6)]  # one subject, 6 stimuli
        ds = _StubDataset(index, n_classes=3)
        model = _ConstantPredictModel(n_classes=3)

        fp = run_fold_inference(
            model, ds, val_subject_ids=[0], batch_size=3, device="cpu", use_amp=False
        )
        # Labels: 0,1,2,0,1,2 — model predicts all 0 → 2/6 correct.
        assert fp.window_acc == pytest.approx(2 / 6)
        assert fp.per_subject_acc[0] == pytest.approx(2 / 6)

    def test_per_subject_acc_uses_only_requested_subjects(self):
        # Dataset has subjects 0, 1, 2 but val_subject_ids only requests {0, 2}.
        index = [(sid, 0, 0) for sid in [0, 1, 2]]
        ds = _StubDataset(index, n_classes=3)
        model = _PerfectModel(ds)

        fp = run_fold_inference(
            model, ds, val_subject_ids=[0, 2], batch_size=3, device="cpu", use_amp=False
        )
        assert set(fp.per_subject_acc.keys()) == {0, 2}
        # Subject 1 doesn't appear in the per_subject dict at all.
        assert 1 not in fp.per_subject_acc
        assert 1 not in fp.per_subject_support

    def test_subject_not_in_dataset_silently_skipped(self):
        # Requested subject 99 has no windows in the dataset.
        index = [(0, 0, 0), (1, 0, 0)]
        ds = _StubDataset(index, n_classes=3)
        model = _PerfectModel(ds)
        fp = run_fold_inference(
            model, ds, val_subject_ids=[0, 1, 99], batch_size=2, device="cpu", use_amp=False
        )
        assert 99 not in fp.per_subject_acc
        assert {0, 1} == set(fp.per_subject_acc.keys())

    def test_order_preserved_across_batches(self):
        # With shuffle=False (the contract), the index order must match
        # batch concatenation order. We verify by checking the recovered
        # subj_ids equal the dataset.index column.
        index = [(sid, stim, 0) for sid in range(4) for stim in range(5)]
        ds = _StubDataset(index, n_classes=5)
        model = _PerfectModel(ds)
        fp = run_fold_inference(
            model, ds, val_subject_ids=[0, 1, 2, 3], batch_size=7, device="cpu", use_amp=False
        )
        expected_subj_ids = np.array([row[0] for row in index], dtype=np.int32)
        np.testing.assert_array_equal(fp.subj_ids, expected_subj_ids)


# ── write_run_summary ────────────────────────────────────────────────────


def _make_fp(
    fold: int, subjects: list[int], n_classes: int = 3, acc: float = 1.0
) -> FoldPredictions:
    """Build a small synthetic FoldPredictions for an aggregator test."""
    rng = np.random.default_rng(fold)
    n_per_subject = 4
    subj_ids = np.repeat(subjects, n_per_subject).astype(np.int32)
    stim_ids = np.tile(np.arange(n_per_subject, dtype=np.int32), len(subjects))
    window_starts = np.zeros_like(stim_ids)
    y_true = rng.integers(0, n_classes, size=len(subj_ids)).astype(np.int32)
    # Predict with controlled accuracy: flip a fraction.
    y_pred = y_true.copy()
    if acc < 1.0:
        flip = rng.random(size=len(y_true)) > acc
        y_pred = np.where(flip, (y_pred + 1) % n_classes, y_pred).astype(np.int32)
    # Softmax probs: very confident on the predicted label.
    y_prob = np.full((len(y_true), n_classes), 0.05, dtype=np.float32)
    y_prob[np.arange(len(y_true)), y_pred] = 0.9
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    return FoldPredictions(
        fold=fold,
        gen_seed=None,
        val_subject_ids=list(subjects),
        val_stimuli=None,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob.astype(np.float32),
        subj_ids=subj_ids,
        stim_ids=stim_ids,
        window_starts=window_starts,
        per_subject_acc={
            int(s): float((y_pred[subj_ids == s] == y_true[subj_ids == s]).mean()) for s in subjects
        },
        per_subject_support={int(s): n_per_subject for s in subjects},
        window_acc=float((y_pred == y_true).mean()),
    )


class TestWriteRunSummary:
    def test_writes_json_and_npz(self, tmp_path: Path):
        fps = [_make_fp(1, [0, 1])]
        json_path = write_run_summary(
            fps,
            approach="jade",
            task="9-class",
            dataset="faced",
            run_stem="test_stem",
            out_root=tmp_path,
        )
        assert json_path.exists()
        assert json_path.suffix == ".json"
        assert json_path.with_suffix(".npz").exists()

    def test_empty_fold_preds_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="empty"):
            write_run_summary(
                [],
                approach="jade",
                task="9-class",
                dataset="faced",
                run_stem="x",
                out_root=tmp_path,
            )

    def test_subject_in_two_folds_raises(self, tmp_path: Path):
        # Cross-subject CV invariant: each subject appears in exactly one fold.
        fps = [_make_fp(1, [0]), _make_fp(2, [0])]  # subject 0 in both
        with pytest.raises(RuntimeError, match="two folds"):
            write_run_summary(
                fps,
                approach="jade",
                task="9-class",
                dataset="faced",
                run_stem="x",
                out_root=tmp_path,
            )

    def test_json_has_expected_schema(self, tmp_path: Path):
        fps = [_make_fp(1, [0, 1]), _make_fp(2, [2, 3])]
        json_path = write_run_summary(
            fps,
            approach="jade",
            task="9-class",
            dataset="faced",
            run_stem="test_stem",
            out_root=tmp_path,
        )
        data = json.loads(json_path.read_text())
        # Top-level keys required by statistical_tests.py and the visualization scripts.
        expected_keys = {
            "approach",
            "task",
            "dataset",
            "run_stem",
            "completed_at",
            "n_folds_run",
            "n_subjects",
            "gen_seed",
            "subject_wise",
            "window_wise_acc",
            "classification_report",
            "per_subject_acc",
            "per_subject_support",
            "folds",
        }
        assert expected_keys.issubset(data.keys())

    def test_per_subject_acc_keys_are_strings_in_json(self, tmp_path: Path):
        # JSON cannot have integer keys; we cast to str for serialization.
        fps = [_make_fp(1, [0, 1])]
        json_path = write_run_summary(
            fps,
            approach="jade",
            task="9-class",
            dataset="faced",
            run_stem="x",
            out_root=tmp_path,
        )
        data = json.loads(json_path.read_text())
        for sid in data["per_subject_acc"].keys():
            assert isinstance(sid, str)
            # And the string can be cast back to int.
            int(sid)

    def test_subject_wise_block_shape(self, tmp_path: Path):
        fps = [_make_fp(1, list(range(10)), acc=0.7)]
        json_path = write_run_summary(
            fps,
            approach="ft",
            task="binary",
            dataset="faced",
            run_stem="x",
            out_root=tmp_path,
        )
        data = json.loads(json_path.read_text())
        sw = data["subject_wise"]
        assert set(sw.keys()) == {"mean_acc", "std_acc", "min_acc", "max_acc"}
        # std with ddof=1 across 10 finite floats
        accs = np.array(list(data["per_subject_acc"].values()), dtype=float)
        np.testing.assert_allclose(sw["mean_acc"], accs.mean())
        np.testing.assert_allclose(sw["std_acc"], accs.std(ddof=1))

    def test_classification_report_shapes(self, tmp_path: Path):
        fps = [_make_fp(1, [0, 1, 2], n_classes=3)]
        json_path = write_run_summary(
            fps,
            approach="jade",
            task="9-class",
            dataset="faced",
            run_stem="x",
            out_root=tmp_path,
        )
        cr = json.loads(json_path.read_text())["classification_report"]
        assert "labels" in cr
        assert "confusion_matrix" in cr
        assert "per_class" in cr
        assert "macro" in cr
        # Confusion matrix is square with side = len(labels)
        cm = np.array(cr["confusion_matrix"])
        assert cm.shape == (len(cr["labels"]), len(cr["labels"]))

    def test_extra_metadata_persists(self, tmp_path: Path):
        fps = [_make_fp(1, [0, 1])]
        json_path = write_run_summary(
            fps,
            approach="jade",
            task="9-class",
            dataset="faced",
            run_stem="x",
            out_root=tmp_path,
            extra_metadata={"alpha": 0.3, "tau": 0.2},
        )
        data = json.loads(json_path.read_text())
        assert data["extra"]["alpha"] == 0.3
        assert data["extra"]["tau"] == 0.2

    def test_generalization_writes_to_separate_folder(self, tmp_path: Path):
        fps = [_make_fp(1, [0, 1])]
        json_path = write_run_summary(
            fps,
            approach="jade",
            task="9-class",
            dataset="faced",
            run_stem="x",
            out_root=tmp_path,
            generalization=True,
            gen_seed=123,
        )
        assert json_path.parent.name == "jade_9-class_generalization"
        data = json.loads(json_path.read_text())
        assert data["gen_seed"] == 123

    def test_npz_has_expected_arrays(self, tmp_path: Path):
        fps = [_make_fp(1, [0, 1])]
        json_path = write_run_summary(
            fps,
            approach="jade",
            task="9-class",
            dataset="faced",
            run_stem="x",
            out_root=tmp_path,
        )
        npz_path = json_path.with_suffix(".npz")
        npz = np.load(npz_path)
        expected = {"y_true", "y_pred", "y_prob", "subj_ids", "stim_ids", "labels"}
        assert expected.issubset(set(npz.files))
        # All per-window arrays have aligned length.
        n = len(npz["y_true"])
        assert len(npz["y_pred"]) == n
        assert len(npz["y_prob"]) == n
        assert len(npz["subj_ids"]) == n
        assert len(npz["stim_ids"]) == n

    def test_output_path_uses_run_stem_unchanged(self, tmp_path: Path):
        # The filename is exactly <run_stem>.json — no double-encoding of gen_seed.
        fps = [_make_fp(1, [0, 1])]
        json_path = write_run_summary(
            fps,
            approach="jade",
            task="9-class",
            dataset="faced",
            run_stem="my_custom_stem",
            out_root=tmp_path,
            generalization=True,
            gen_seed=123,
        )
        assert json_path.name == "my_custom_stem.json"


# ── Round-trip: run_fold_inference → write_run_summary ─────────────────────


class TestRoundTrip:
    def test_e2e_perfect_run(self, tmp_path: Path):
        # Build two folds, run inference on each, write the summary, read it back.
        index_f1 = [(0, stim, 0) for stim in range(3)] + [(1, stim, 0) for stim in range(3)]
        index_f2 = [(2, stim, 0) for stim in range(3)] + [(3, stim, 0) for stim in range(3)]
        ds1 = _StubDataset(index_f1, n_classes=3)
        ds2 = _StubDataset(index_f2, n_classes=3)
        m1 = _PerfectModel(ds1)
        m2 = _PerfectModel(ds2)

        fp1 = run_fold_inference(
            m1, ds1, val_subject_ids=[0, 1], batch_size=3, device="cpu", use_amp=False
        )
        fp2 = run_fold_inference(
            m2, ds2, val_subject_ids=[2, 3], batch_size=3, device="cpu", use_amp=False
        )
        fp1.fold = 1
        fp2.fold = 2

        json_path = write_run_summary(
            [fp1, fp2],
            approach="jade",
            task="9-class",
            dataset="faced",
            run_stem="e2e",
            out_root=tmp_path,
        )
        data = json.loads(json_path.read_text())
        # 4 subjects, all perfect accuracy.
        assert data["n_subjects"] == 4
        assert data["subject_wise"]["mean_acc"] == 1.0
        assert data["window_wise_acc"] == 1.0
        assert set(int(k) for k in data["per_subject_acc"]) == {0, 1, 2, 3}
