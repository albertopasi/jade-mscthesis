"""
inference_subject_wise.py — Post-hoc inference on saved per-fold checkpoints.

Reproduces the val split for each fold (from fold_meta.json if present, else
from get_kfold_splits with the canonical seed), runs deterministic inference
with the saved model, and aggregates per-subject accuracies + pooled
confusion matrix + per-class P/R/F1.

Each subject appears in exactly one val fold in cross-subject 10-fold CV,
so pooling per-subject accuracies across folds gives one accuracy per
subject in the dataset. The reported headline std is the std across those
N_subjects accuracies (comparable to subject-wise std in literature).

Usage:
    uv run python -m src.inference.inference_subject_wise \\
        --approach lp --task 9-class \\
        --ckpt-root outputs/lp_checkpoints \\
        --run-stem lp_faced_v2_9-class_w10s10_pool_no_official_nomixup

    uv run python -m src.inference.inference_subject_wise \\
        --approach ft --task 9-class \\
        --ckpt-root outputs/ft_checkpoints \\
        --run-stem ft_faced_9-class_w10s10_pool_no_r16_b256_lr0.0004_nomixup_fullft

REVE paths default to local models/reve_pretrained_original/. Override
with --reve-base / --reve-pos if needed.

Output JSON written to: <ckpt-root>/inference_<run-stem>.json
"""

from __future__ import annotations

import argparse
import datetime
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.approaches.fine_tuning.model import ReveClassifierFT
from src.approaches.jade.model import ReveClassifierJADE
from src.approaches.linear_probing.model import ReveClassifierLP
from src.approaches.shared.reve import get_channel_names, load_reve_and_positions
from src.datasets.faced_dataset import FACEDWindowDataset
from src.datasets.folds import get_all_subjects, get_kfold_splits

# ── Constants ──────────────────────────────────────────────────────────────────
SAMPLING_RATE = 200
DATA_ROOTS = {
    "faced": Path("data/FACED/preprocessed_v2"),
}
N_CHANNELS = {"faced": 32}
N_CLASSES = {"9-class": 9, "binary": 2}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--approach", choices=["lp", "ft", "jade"], required=True)
    p.add_argument("--task", choices=["9-class", "binary"], required=True)
    p.add_argument("--dataset", choices=["faced", "thu-ep"], default="faced")
    p.add_argument("--ckpt-root", type=Path, required=True,
                   help="Directory containing the per-fold checkpoint dirs.")
    p.add_argument("--run-stem", type=str, required=True,
                   help="Run-name stem; per-fold dirs are <stem>_fold_{1..10}.")
    p.add_argument("--window", type=int, default=10, help="Window size in seconds (default 10).")
    p.add_argument("--stride", type=int, default=10, help="Stride in seconds (default 10).")
    p.add_argument("--pooling", choices=["no", "last", "last_avg"], default="no")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--reve-base", type=Path,
                   default=Path("models/reve_pretrained_original/reve-base"))
    p.add_argument("--reve-pos", type=Path,
                   default=Path("models/reve_pretrained_original/reve-positions"))
    p.add_argument("--n-folds", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # JADE/FT extras (only used when building the model)
    p.add_argument("--lora-rank", type=int, default=16,
                   help="LoRA rank (FT/JADE — only matters when reading non-fullft checkpoints).")
    p.add_argument("--out-path", type=Path, default=None,
                   help="Output JSON path. Default: <ckpt-root>/inference_<run-stem>.json")
    return p.parse_args()


def build_model(
    approach: str,
    task: str,
    dataset: str,
    pooling: str,
    window_size: int,
    reve_model,
    pos_tensor,
):
    """Instantiate the classifier matching what was trained."""
    n_classes = N_CLASSES[task]
    n_channels = N_CHANNELS[dataset]
    if approach == "lp":
        return ReveClassifierLP(
            reve_model=reve_model, pos_tensor=pos_tensor,
            n_classes=n_classes, n_channels=n_channels,
            window_size=window_size, pooling=pooling, dropout=0.0,
        )
    if approach == "ft":
        return ReveClassifierFT(
            reve_model=reve_model, pos_tensor=pos_tensor,
            n_classes=n_classes, n_channels=n_channels,
            window_size=window_size, pooling=pooling, dropout=0.0,
        )
    if approach == "jade":
        return ReveClassifierJADE(
            reve_model=reve_model, pos_tensor=pos_tensor,
            n_classes=n_classes, n_channels=n_channels,
            window_size=window_size, pooling=pooling, dropout=0.0,
        )
    raise ValueError(approach)


def load_weights(model: torch.nn.Module, ckpt_dir: Path, approach: str) -> None:
    """Load weights into the model in-place. Different approaches save different files."""
    if approach == "lp":
        # LP saves only trainable params: cls_query_token + linear_head
        state = torch.load(ckpt_dir / "classifier_weights.pt", map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        # missing should include all reve.* params (frozen, weights from REVE itself); unexpected should be empty
        if unexpected:
            raise RuntimeError(f"Unexpected keys in LP ckpt: {unexpected[:5]}...")
        return
    if approach in ("ft", "jade"):
        # Full-FT path: single full_model.pt with the entire (non-projection-head) state dict
        full_path = ckpt_dir / "full_model.pt"
        if full_path.exists():
            state = torch.load(full_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
            return
        # LoRA path: adapter + head separately. Not supported in this script yet —
        # would need to wrap encoder with peft before loading. The current re-runs
        # are full-ft only, so we don't need this branch right now.
        raise NotImplementedError(
            f"Only full_model.pt loading implemented; {ckpt_dir} has no full_model.pt"
        )
    raise ValueError(approach)


def get_val_subjects_for_fold(
    ckpt_dir: Path, fold_idx: int, all_subjects: list[int], n_folds: int
) -> list[int]:
    """Recover the val subject list for a fold.

    Prefer fold_meta.json if present (FT/JADE save it). Otherwise reconstruct
    from get_kfold_splits with the canonical seed — same result as long as
    `get_all_subjects` returns the same ordering used at training time.
    """
    meta = ckpt_dir / "fold_meta.json"
    if meta.exists():
        return json.loads(meta.read_text())["val_subject_ids"]
    folds = get_kfold_splits(all_subjects, n_folds=n_folds)
    _, val_idx = folds[fold_idx - 1]  # fold_idx is 1-based
    return [all_subjects[i] for i in val_idx]


@torch.no_grad()
def infer_one_fold(
    model: torch.nn.Module,
    val_ds: FACEDWindowDataset,
    val_subjects: list[int],
    batch_size: int,
    device: str,
) -> dict:
    """Run inference on a fold's val set, returning per-window arrays.

    Critical: shuffle=False + num_workers=0 so that batch order matches
    `val_ds.index`, letting us recover the subject_id per window via index lookup.
    """
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    model.eval()
    y_true_all, y_pred_all = [], []
    for eeg, label in loader:
        eeg = eeg.to(device, non_blocking=True)
        logits = model(eeg)
        preds = logits.argmax(dim=-1).cpu().numpy()
        y_pred_all.append(preds)
        y_true_all.append(label.numpy())
    y_true = np.concatenate(y_true_all).astype(int)
    y_pred = np.concatenate(y_pred_all).astype(int)
    # Subject id per window: dataset.index[i] = (sid, stim_idx, window_start)
    subj_ids = np.array([val_ds.index[i][0] for i in range(len(val_ds))], dtype=int)
    assert len(subj_ids) == len(y_true), f"index mismatch {len(subj_ids)} vs {len(y_true)}"
    # Per-subject accuracy
    per_subject_acc: dict[int, float] = {}
    per_subject_support: dict[int, int] = {}
    for sid in val_subjects:
        mask = subj_ids == sid
        if mask.sum() == 0:
            continue
        per_subject_acc[sid] = float((y_pred[mask] == y_true[mask]).mean())
        per_subject_support[sid] = int(mask.sum())
    return {
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "subj_ids": subj_ids.tolist(),
        "per_subject_acc": per_subject_acc,
        "per_subject_support": per_subject_support,
        "fold_acc_window": float((y_pred == y_true).mean()),
    }


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    # Switch CWD to project root so relative paths in DATA_ROOTS work
    data_root = (project_root / DATA_ROOTS[args.dataset]).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    window_size = args.window * SAMPLING_RATE
    stride = args.stride * SAMPLING_RATE

    print(f"\n=== Subject-wise inference ===")
    print(f"Approach: {args.approach}  |  Task: {args.task}  |  Dataset: {args.dataset}")
    print(f"Ckpt root: {args.ckpt_root}")
    print(f"Run stem:  {args.run_stem}")
    print(f"REVE base: {args.reve_base}")
    print(f"Window: {args.window}s, stride: {args.stride}s, pooling: {args.pooling}\n")

    # Load REVE once (shared across all folds; frozen for LP, finetuned weights
    # get loaded on top per-fold for FT/JADE)
    channel_names = get_channel_names(args.dataset)
    reve_model, pos_tensor = load_reve_and_positions(
        channel_names, device=args.device,
        reve_model_path=args.reve_base, reve_pos_path=args.reve_pos,
    )

    all_subjects = get_all_subjects(args.dataset)

    fold_results: list[dict] = []
    pooled_per_subject_acc: dict[int, float] = {}
    pooled_per_subject_support: dict[int, int] = {}
    pooled_y_true: list[int] = []
    pooled_y_pred: list[int] = []

    for fold_idx in range(1, args.n_folds + 1):
        ckpt_dir = args.ckpt_root / f"{args.run_stem}_fold_{fold_idx}"
        if not ckpt_dir.exists():
            print(f"[fold {fold_idx}] SKIP — {ckpt_dir} not found")
            continue

        val_subjects = get_val_subjects_for_fold(ckpt_dir, fold_idx, all_subjects, args.n_folds)
        print(f"[fold {fold_idx}] val_subjects ({len(val_subjects)}): {val_subjects}")

        # Build val dataset
        val_ds = FACEDWindowDataset(
            subject_ids=val_subjects,
            task_mode=args.task,
            data_root=data_root,
            window_size=window_size,
            stride=stride,
        )

        # Build model fresh per fold + load weights
        model = build_model(
            args.approach, args.task, args.dataset, args.pooling,
            window_size, reve_model, pos_tensor,
        )
        load_weights(model, ckpt_dir, args.approach)
        model.to(args.device)

        out = infer_one_fold(model, val_ds, val_subjects, args.batch_size, args.device)
        fold_acc_macro = float(np.mean(list(out["per_subject_acc"].values()))) if out["per_subject_acc"] else float("nan")
        print(f"[fold {fold_idx}] window_acc={out['fold_acc_window']:.4f}  "
              f"macro_subject_acc={fold_acc_macro:.4f}  n_subjects={len(out['per_subject_acc'])}")

        fold_results.append({
            "fold": fold_idx,
            "val_subjects": val_subjects,
            "window_acc": out["fold_acc_window"],
            "macro_subject_acc": fold_acc_macro,
            "per_subject_acc": out["per_subject_acc"],
        })

        for sid, acc in out["per_subject_acc"].items():
            if sid in pooled_per_subject_acc:
                raise RuntimeError(f"Subject {sid} appeared in two folds — split overlap bug")
            pooled_per_subject_acc[sid] = acc
            pooled_per_subject_support[sid] = out["per_subject_support"][sid]

        pooled_y_true.extend(out["y_true"])
        pooled_y_pred.extend(out["y_pred"])

        # Free memory
        del model, val_ds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Aggregate ────────────────────────────────────────────────────────────
    subject_accs = np.array(list(pooled_per_subject_acc.values()))
    n_subjects = len(subject_accs)
    print(f"\n=== Pooled across {len(fold_results)} folds ===")
    print(f"  N subjects: {n_subjects}")
    print(f"  Subject-wise acc: mean={subject_accs.mean():.4f}  std={subject_accs.std(ddof=1):.4f}")
    print(f"  Window-wise acc:  {(np.array(pooled_y_pred) == np.array(pooled_y_true)).mean():.4f}")

    # Confusion matrix + per-class P/R/F1
    labels = sorted(set(pooled_y_true) | set(pooled_y_pred))
    cm = confusion_matrix(pooled_y_true, pooled_y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        pooled_y_true, pooled_y_pred, labels=labels, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        pooled_y_true, pooled_y_pred, labels=labels, average="macro", zero_division=0
    )

    print(f"\n  Per-class:")
    print(f"  {'class':>6}  {'P':>8}  {'R':>8}  {'F1':>8}  {'support':>8}")
    for i, lbl in enumerate(labels):
        print(f"  {lbl:>6}  {precision[i]:>8.4f}  {recall[i]:>8.4f}  {f1[i]:>8.4f}  {int(support[i]):>8d}")
    print(f"  {'macro':>6}  {macro_p:>8.4f}  {macro_r:>8.4f}  {macro_f1:>8.4f}")

    # ── Write JSON ───────────────────────────────────────────────────────────
    out_path = args.out_path or (args.ckpt_root / f"inference_{args.run_stem}.json")
    summary = {
        "approach": args.approach,
        "task": args.task,
        "dataset": args.dataset,
        "run_stem": args.run_stem,
        "ckpt_root": str(args.ckpt_root),
        "completed_at": datetime.datetime.now().isoformat(),
        "n_folds_run": len(fold_results),
        "n_subjects": n_subjects,
        "subject_wise": {
            "mean_acc": float(subject_accs.mean()),
            "std_acc": float(subject_accs.std(ddof=1)),
            "min_acc": float(subject_accs.min()),
            "max_acc": float(subject_accs.max()),
        },
        "window_wise_acc": float((np.array(pooled_y_pred) == np.array(pooled_y_true)).mean()),
        "classification_report": {
            "labels": [int(x) for x in labels],
            "confusion_matrix": cm.tolist(),
            "per_class": {
                "precision": [round(float(x), 4) for x in precision],
                "recall":    [round(float(x), 4) for x in recall],
                "f1":        [round(float(x), 4) for x in f1],
                "support":   [int(x) for x in support],
            },
            "macro": {
                "precision": round(float(macro_p), 4),
                "recall":    round(float(macro_r), 4),
                "f1":        round(float(macro_f1), 4),
            },
        },
        "per_subject_acc": {str(sid): acc for sid, acc in pooled_per_subject_acc.items()},
        "per_subject_support": {str(sid): n for sid, n in pooled_per_subject_support.items()},
        "folds": [
            {"fold": r["fold"], "val_subjects": r["val_subjects"],
             "window_acc": r["window_acc"], "macro_subject_acc": r["macro_subject_acc"]}
            for r in fold_results
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
