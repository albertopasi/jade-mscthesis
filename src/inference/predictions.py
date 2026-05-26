"""
predictions.py — Fold-level inference primitive.

Single responsibility: given a trained model and a val dataset, run forward
inference and return a FoldPredictions dataclass holding per-window arrays
(y_true / y_pred / y_prob / subj_ids / stim_ids / window_starts) plus
derived per-subject accuracies.

Used by:
  - src.inference.inference_subject_wise (post-hoc, loads weights from disk)
  - src.approaches.{jade,fine_tuning}.train_* (in-training, model already in memory)

Both call sites produce the same FoldPredictions, which then feeds into
src.inference.aggregate.write_run_summary().
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class FoldPredictions:
    """All per-window arrays + per-subject derived metrics for one fold.

    The (subj_ids, stim_ids, window_starts) trio is recovered from
    `val_ds.index` (each entry is `(sid, stim_idx, window_start)`), which
    means the val DataLoader must be unshuffled and not sharded.
    """

    fold: int | None = None
    gen_seed: int | None = None
    val_subject_ids: list[int] = field(default_factory=list)
    val_stimuli: list[int] | None = None  # held-out stimulus indices (generalization mode)

    # Per-window arrays (aligned, length = N_windows)
    y_true: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    y_pred: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    y_prob: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))
    subj_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    stim_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    window_starts: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))

    # Derived per-subject metrics
    per_subject_acc: dict[int, float] = field(default_factory=dict)
    per_subject_support: dict[int, int] = field(default_factory=dict)
    window_acc: float = float("nan")


@torch.no_grad()
def run_fold_inference(
    model: nn.Module,
    val_ds: Dataset,
    val_subject_ids: list[int],
    *,
    batch_size: int,
    device: str,
    use_amp: bool = True,
    num_workers: int = 0,
) -> FoldPredictions:
    """Run forward inference on val_ds and return a populated FoldPredictions.

    `val_ds` must expose `.index` as a list of `(subject_id, stimulus_idx,
    window_start)` tuples (the contract enforced by EEGWindowDataset). This
    is how we recover per-window subject/stimulus identity without
    propagating them through the model.

    The DataLoader is built with shuffle=False so batch order matches
    `val_ds.index` 1:1.
    """
    if not hasattr(val_ds, "index"):
        raise TypeError(
            f"val_ds must expose a `.index` attribute mapping window position "
            f"to (subject_id, stimulus_idx, window_start); got {type(val_ds).__name__}"
        )

    loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model.eval()
    device_type = "cuda" if "cuda" in device else "cpu"

    y_true_chunks: list[np.ndarray] = []
    y_pred_chunks: list[np.ndarray] = []
    y_prob_chunks: list[np.ndarray] = []

    for batch in loader:
        eeg = batch[0].to(device, non_blocking=True)
        label = batch[1]

        with torch.autocast(device_type=device_type, enabled=use_amp, dtype=torch.float16):
            with torch.inference_mode():
                logits = model(eeg)

        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()

        y_true_chunks.append(label.numpy().astype(np.int32))
        y_pred_chunks.append(preds.astype(np.int32))
        y_prob_chunks.append(probs.astype(np.float32))

    y_true = np.concatenate(y_true_chunks)
    y_pred = np.concatenate(y_pred_chunks)
    y_prob = np.concatenate(y_prob_chunks, axis=0)

    # Recover per-window identifiers from the dataset's flat index.
    index_array = np.asarray(val_ds.index, dtype=np.int32)
    if len(index_array) != len(y_true):
        raise RuntimeError(
            f"index/length mismatch: val_ds.index has {len(index_array)} rows but "
            f"inference produced {len(y_true)} predictions"
        )
    subj_ids = index_array[:, 0]
    stim_ids = index_array[:, 1]
    window_starts = index_array[:, 2]

    # Per-subject accuracy + support.
    per_subject_acc: dict[int, float] = {}
    per_subject_support: dict[int, int] = {}
    for sid in val_subject_ids:
        mask = subj_ids == sid
        n_sid = int(mask.sum())
        if n_sid == 0:
            continue
        per_subject_acc[int(sid)] = float((y_pred[mask] == y_true[mask]).mean())
        per_subject_support[int(sid)] = n_sid

    window_acc = float((y_pred == y_true).mean()) if len(y_true) else float("nan")

    return FoldPredictions(
        val_subject_ids=list(val_subject_ids),
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        subj_ids=subj_ids,
        stim_ids=stim_ids,
        window_starts=window_starts,
        per_subject_acc=per_subject_acc,
        per_subject_support=per_subject_support,
        window_acc=window_acc,
    )
