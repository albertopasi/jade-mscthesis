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

This script is the post-hoc entry point. The training pipelines
(train_jade.py, train_ft.py) produce the same JSON+NPZ output without
needing checkpoints when --generalization is used. Both paths share
src.inference.predictions.run_fold_inference and
src.inference.aggregate.write_run_summary so the schema lives in one place.

Usage:
    uv run python -m src.inference.inference_subject_wise \\
        --approach lp --task 9-class \\
        --ckpt-root outputs/lp_checkpoints \\
        --run-stem lp_faced_v2_9-class_w10s10_pool_no_official_nomixup

    uv run python -m src.inference.inference_subject_wise \\
        --approach ft --task 9-class \\
        --ckpt-root outputs/ft_checkpoints \\
        --run-stem ft_faced_9-class_w10s10_pool_no_r16_b256_lr0.0004_nomixup_fullft
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.approaches.fine_tuning.model import ReveClassifierFT
from src.approaches.jade.model import ReveClassifierJADE
from src.approaches.linear_probing.model import ReveClassifierLP
from src.approaches.shared.reve import get_channel_names, load_reve_and_positions
from src.datasets.faced_dataset import FACEDWindowDataset
from src.datasets.folds import get_all_subjects, get_kfold_splits
from src.inference.aggregate import write_run_summary
from src.inference.predictions import FoldPredictions, run_fold_inference

# ── Constants ──────────────────────────────────────────────────────────────────
SAMPLING_RATE = 200
DATA_ROOTS = {
    "faced": Path("data/FACED/preprocessed_v2"),
}
N_CHANNELS = {"faced": 32}
N_CLASSES = {"9-class": 9, "binary": 2}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--approach", choices=["lp", "ft", "jade"], required=True)
    p.add_argument("--task", choices=["9-class", "binary"], required=True)
    p.add_argument("--dataset", choices=["faced", "thu-ep"], default="faced")
    p.add_argument(
        "--ckpt-root",
        type=Path,
        required=True,
        help="Directory containing the per-fold checkpoint dirs.",
    )
    p.add_argument(
        "--run-stem",
        type=str,
        required=True,
        help="Run-name stem; per-fold dirs are <stem>_fold_{1..10}.",
    )
    p.add_argument("--window", type=int, default=10, help="Window size in seconds (default 10).")
    p.add_argument("--stride", type=int, default=10, help="Stride in seconds (default 10).")
    p.add_argument("--pooling", choices=["no", "last", "last_avg"], default="no")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--reve-base", type=Path, default=Path("models/reve_pretrained_original/reve-base")
    )
    p.add_argument(
        "--reve-pos", type=Path, default=Path("models/reve_pretrained_original/reve-positions")
    )
    p.add_argument("--n-folds", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank (FT/JADE — only matters when reading non-fullft checkpoints).",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output root. Default: main-results/ at project root.",
    )
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
            reve_model=reve_model,
            pos_tensor=pos_tensor,
            n_classes=n_classes,
            n_channels=n_channels,
            window_size=window_size,
            pooling=pooling,
            dropout=0.0,
        )
    if approach == "ft":
        return ReveClassifierFT(
            reve_model=reve_model,
            pos_tensor=pos_tensor,
            n_classes=n_classes,
            n_channels=n_channels,
            window_size=window_size,
            pooling=pooling,
            dropout=0.0,
        )
    if approach == "jade":
        return ReveClassifierJADE(
            reve_model=reve_model,
            pos_tensor=pos_tensor,
            n_classes=n_classes,
            n_channels=n_channels,
            window_size=window_size,
            pooling=pooling,
            dropout=0.0,
        )
    raise ValueError(approach)


def load_weights(model: torch.nn.Module, ckpt_dir: Path, approach: str) -> None:
    """Load weights into the model in-place. Different approaches save different files."""
    if approach == "lp":
        # LP saves only trainable params: cls_query_token + linear_head
        state = torch.load(ckpt_dir / "classifier_weights.pt", map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys in LP ckpt: {unexpected[:5]}...")
        return
    if approach in ("ft", "jade"):
        full_path = ckpt_dir / "full_model.pt"
        if full_path.exists():
            state = torch.load(full_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
            return
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


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    data_root = (project_root / DATA_ROOTS[args.dataset]).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    window_size = args.window * SAMPLING_RATE
    stride = args.stride * SAMPLING_RATE

    print("\n=== Subject-wise inference ===")
    print(f"Approach: {args.approach}  |  Task: {args.task}  |  Dataset: {args.dataset}")
    print(f"Ckpt root: {args.ckpt_root}")
    print(f"Run stem:  {args.run_stem}")
    print(f"REVE base: {args.reve_base}")
    print(f"Window: {args.window}s, stride: {args.stride}s, pooling: {args.pooling}\n")

    channel_names = get_channel_names(args.dataset)
    reve_model, pos_tensor = load_reve_and_positions(
        channel_names,
        device=args.device,
        reve_model_path=args.reve_base,
        reve_pos_path=args.reve_pos,
    )

    all_subjects = get_all_subjects(args.dataset)

    fold_preds: list[FoldPredictions] = []

    for fold_idx in range(1, args.n_folds + 1):
        ckpt_dir = args.ckpt_root / f"{args.run_stem}_fold_{fold_idx}"
        if not ckpt_dir.exists():
            print(f"[fold {fold_idx}] SKIP — {ckpt_dir} not found")
            continue

        val_subjects = get_val_subjects_for_fold(ckpt_dir, fold_idx, all_subjects, args.n_folds)
        print(f"[fold {fold_idx}] val_subjects ({len(val_subjects)}): {val_subjects}")

        val_ds = FACEDWindowDataset(
            subject_ids=val_subjects,
            task_mode=args.task,
            data_root=data_root,
            window_size=window_size,
            stride=stride,
        )

        model = build_model(
            args.approach,
            args.task,
            args.dataset,
            args.pooling,
            window_size,
            reve_model,
            pos_tensor,
        )
        load_weights(model, ckpt_dir, args.approach)
        model.to(args.device)

        fp = run_fold_inference(
            model,
            val_ds,
            val_subjects,
            batch_size=args.batch_size,
            device=args.device,
            use_amp=False,  # post-hoc inference: prefer determinism over speed
        )
        fp.fold = fold_idx
        # Post-hoc inference reproduces the standard k-fold val set, so no held-out stimuli.
        fold_preds.append(fp)

        macro_subject_acc = (
            float(np.mean(list(fp.per_subject_acc.values())))
            if fp.per_subject_acc
            else float("nan")
        )
        print(
            f"[fold {fold_idx}] window_acc={fp.window_acc:.4f}  "
            f"macro_subject_acc={macro_subject_acc:.4f}  n_subjects={len(fp.per_subject_acc)}"
        )

        del model, val_ds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not fold_preds:
        raise RuntimeError("No fold checkpoints found — nothing to aggregate.")

    out_root = args.out_root or (project_root / "main-results")
    json_path = write_run_summary(
        fold_preds,
        approach=args.approach,
        task=args.task,
        dataset=args.dataset,
        run_stem=args.run_stem,
        out_root=out_root,
        generalization=False,
    )

    # Brief stdout summary for the operator.
    subject_accs = np.array(
        [a for fp in fold_preds for a in fp.per_subject_acc.values()], dtype=float
    )
    print(f"\n=== Pooled across {len(fold_preds)} folds ===")
    print(f"  N subjects: {len(subject_accs)}")
    if len(subject_accs) > 1:
        print(
            f"  Subject-wise acc: mean={subject_accs.mean():.4f}  "
            f"std={subject_accs.std(ddof=1):.4f}"
        )
    print(f"\nSaved → {json_path}")
    print(f"Saved → {json_path.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
