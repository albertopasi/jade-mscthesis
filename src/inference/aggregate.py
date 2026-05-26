"""
aggregate.py — Pooled metrics + JSON/NPZ writer for a full run.

Single responsibility: take a list of FoldPredictions covering all folds of
a (method, task) run, compute the pooled descriptive statistics + the same
classification report that inference_subject_wise.py used to produce, and
write the result to disk as `<stem>.json` + `<stem>.npz`.

This function is the **only** place that knows the on-disk schema used by
downstream tools (statistical_tests.py, plot_inference.py). Both
inference_subject_wise.py and the in-training generalization path go
through it.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

from src.inference.predictions import FoldPredictions


def _pooled_macro_auroc(
    y_true: np.ndarray, y_prob: np.ndarray, labels: list[int]
) -> tuple[float, list[float]]:
    """Macro one-vs-rest AUROC + per-class AUROC. Returns NaN on failure."""
    n_classes = y_prob.shape[1]
    per_class: list[float] = []
    for i in range(n_classes):
        y_bin = (y_true == labels[i]).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            per_class.append(float("nan"))
        else:
            per_class.append(float(roc_auc_score(y_bin, y_prob[:, i])))

    try:
        if n_classes == 2:
            macro = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            macro = float(
                roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="macro", labels=labels
                )
            )
    except ValueError:
        macro = float("nan")
    return macro, per_class


def _build_summary_dict(
    *,
    approach: str,
    task: str,
    dataset: str,
    run_stem: str,
    fold_preds: list[FoldPredictions],
    pooled_per_subject_acc: dict[int, float],
    pooled_per_subject_support: dict[int, int],
    pooled_y_true: np.ndarray,
    pooled_y_pred: np.ndarray,
    pooled_y_prob: np.ndarray,
    gen_seed: int | None,
    extra_metadata: dict | None,
) -> dict:
    """Compose the JSON-serializable summary dict (matches the legacy schema)."""
    subject_accs = np.array(list(pooled_per_subject_acc.values()), dtype=float)
    n_subjects = len(subject_accs)

    labels = sorted(set(pooled_y_true.tolist()) | set(pooled_y_pred.tolist()))
    cm = confusion_matrix(pooled_y_true, pooled_y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        pooled_y_true, pooled_y_pred, labels=labels, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        pooled_y_true, pooled_y_pred, labels=labels, average="macro", zero_division=0
    )
    macro_auroc, auroc_per_class = _pooled_macro_auroc(pooled_y_true, pooled_y_prob, labels)

    window_acc = float((pooled_y_pred == pooled_y_true).mean()) if len(pooled_y_true) else float("nan")

    summary: dict = {
        "approach": approach,
        "task": task,
        "dataset": dataset,
        "run_stem": run_stem,
        "completed_at": datetime.datetime.now().isoformat(),
        "n_folds_run": len(fold_preds),
        "n_subjects": n_subjects,
        "gen_seed": gen_seed,
        "subject_wise": {
            "mean_acc": float(subject_accs.mean()) if n_subjects else float("nan"),
            "std_acc": float(subject_accs.std(ddof=1)) if n_subjects > 1 else float("nan"),
            "min_acc": float(subject_accs.min()) if n_subjects else float("nan"),
            "max_acc": float(subject_accs.max()) if n_subjects else float("nan"),
        },
        "window_wise_acc": window_acc,
        "classification_report": {
            "labels": [int(x) for x in labels],
            "confusion_matrix": cm.tolist(),
            "per_class": {
                "precision": [round(float(x), 4) for x in precision],
                "recall": [round(float(x), 4) for x in recall],
                "f1": [round(float(x), 4) for x in f1],
                "support": [int(x) for x in support],
            },
            "macro": {
                "precision": round(float(macro_p), 4),
                "recall": round(float(macro_r), 4),
                "f1": round(float(macro_f1), 4),
                "auroc": round(macro_auroc, 4),
            },
            "auroc_per_class": [round(a, 4) for a in auroc_per_class],
        },
        "per_subject_acc": {str(sid): acc for sid, acc in pooled_per_subject_acc.items()},
        "per_subject_support": {str(sid): n for sid, n in pooled_per_subject_support.items()},
        "folds": [
            {
                "fold": fp.fold,
                "gen_seed": fp.gen_seed,
                "val_subjects": fp.val_subject_ids,
                "val_stimuli": fp.val_stimuli,
                "window_acc": fp.window_acc,
                "macro_subject_acc": (
                    float(np.mean(list(fp.per_subject_acc.values())))
                    if fp.per_subject_acc
                    else float("nan")
                ),
            }
            for fp in fold_preds
        ],
    }
    if extra_metadata:
        summary["extra"] = extra_metadata
    return summary


def write_run_summary(
    fold_preds: list[FoldPredictions],
    *,
    approach: str,
    task: str,
    dataset: str,
    run_stem: str,
    out_root: Path,
    extra_metadata: dict | None = None,
    gen_seed: int | None = None,
    generalization: bool = False,
) -> Path:
    """Pool fold-level predictions, compute summary metrics, write JSON + NPZ.

    The caller is responsible for choosing a `run_stem` that uniquely
    identifies the run; in generalization mode the seed is expected to
    already be embedded in the stem (e.g. via ``cfg.run_name_stem(gen_seed)``,
    which produces ``..._gen_s{seed}``). The `gen_seed` argument is recorded
    in the JSON body for downstream tooling but is **not** appended to the
    filename — that would double-encode it.

    Output paths:
        out_root/<approach>_<task>[_generalization]/<run_stem>.json
        out_root/<approach>_<task>[_generalization]/<run_stem>.npz

    Returns the JSON path.
    """
    if not fold_preds:
        raise ValueError("fold_preds is empty — nothing to aggregate.")

    # Pool per-subject accuracies (each subject appears in exactly one fold).
    pooled_per_subject_acc: dict[int, float] = {}
    pooled_per_subject_support: dict[int, int] = {}
    for fp in fold_preds:
        for sid, acc in fp.per_subject_acc.items():
            if sid in pooled_per_subject_acc:
                raise RuntimeError(
                    f"Subject {sid} appeared in two folds — cross-subject CV split overlap bug."
                )
            pooled_per_subject_acc[sid] = acc
            pooled_per_subject_support[sid] = fp.per_subject_support[sid]

    # Pool per-window arrays.
    pooled_y_true = np.concatenate([fp.y_true for fp in fold_preds]).astype(np.int32)
    pooled_y_pred = np.concatenate([fp.y_pred for fp in fold_preds]).astype(np.int32)
    pooled_y_prob = np.concatenate([fp.y_prob for fp in fold_preds], axis=0).astype(np.float32)
    pooled_subj_ids = np.concatenate([fp.subj_ids for fp in fold_preds]).astype(np.int32)
    pooled_stim_ids = np.concatenate([fp.stim_ids for fp in fold_preds]).astype(np.int32)

    # Resolve output paths.
    folder = f"{approach}_{task}" + ("_generalization" if generalization else "")
    out_dir = out_root / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{run_stem}.json"
    npz_path = out_dir / f"{run_stem}.npz"

    # Build + write JSON.
    summary = _build_summary_dict(
        approach=approach,
        task=task,
        dataset=dataset,
        run_stem=run_stem,
        fold_preds=fold_preds,
        pooled_per_subject_acc=pooled_per_subject_acc,
        pooled_per_subject_support=pooled_per_subject_support,
        pooled_y_true=pooled_y_true,
        pooled_y_pred=pooled_y_pred,
        pooled_y_prob=pooled_y_prob,
        gen_seed=gen_seed,
        extra_metadata=extra_metadata,
    )
    json_path.write_text(json.dumps(summary, indent=2))

    labels = sorted(set(pooled_y_true.tolist()) | set(pooled_y_pred.tolist()))
    np.savez_compressed(
        npz_path,
        y_true=pooled_y_true,
        y_pred=pooled_y_pred,
        y_prob=pooled_y_prob,
        subj_ids=pooled_subj_ids,
        stim_ids=pooled_stim_ids,
        labels=np.array(labels, dtype=np.int32),
    )

    return json_path
