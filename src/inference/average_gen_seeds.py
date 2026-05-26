"""
average_gen_seeds.py — Combine multiple per-seed generalization summaries.

Each generalization training run writes one summary JSON+NPZ per seed under
`main-results/<approach>_<task>_generalization/`. After all seeds finish,
this script:

  1. Globs the matching per-seed JSONs.
  2. For each subject, averages their per-seed accuracies into a single
     subject-level accuracy (subject still the unit of analysis, N = 123).
  3. Re-pools predictions across seeds for the macro confusion matrix and
     per-class P / R / F1 / AUROC.
  4. Writes a combined summary JSON+NPZ alongside the per-seed inputs.

The combined JSON has the same schema as the per-seed JSONs (so
`statistical_tests.py` can consume it unchanged). Two additions:
  - `gen_seeds`: list of seeds that contributed.
  - `n_seeds_per_subject`: how many seeds each subject appeared in.

Usage:
    uv run python -m src.inference.average_gen_seeds \\
        --approach jade --task 9-class

    # Explicit run-stem prefix (when multiple HP configs coexist in the folder)
    uv run python -m src.inference.average_gen_seeds \\
        --approach jade --task 9-class \\
        --stem-prefix jade_faced_9-class_w10s10_pool_no_r16_a0.3_t0.2_context_b256_lr0.0004_fullft_gen
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "main-results"

SEED_SUFFIX_RE = re.compile(r"_gen_s(\d+)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--approach", choices=["lp", "ft", "jade"], required=True)
    p.add_argument("--task", choices=["9-class", "binary"], required=True)
    p.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Root containing <approach>_<task>_generalization/ (default: {DEFAULT_RESULTS_ROOT}).",
    )
    p.add_argument(
        "--stem-prefix",
        type=str,
        default=None,
        help="Optional prefix to disambiguate when multiple HP configs share the folder. "
        "Matched against the per-seed stem ending in `_gen_s<seed>`.",
    )
    p.add_argument(
        "--out-name",
        type=str,
        default=None,
        help="Output filename stem (without extension). Default: <stem>_gen_avg.",
    )
    return p.parse_args()


def find_per_seed_jsons(folder: Path, stem_prefix: str | None) -> dict[int, Path]:
    """Glob `<folder>/*_gen_s<seed>.json` and return {seed: path}.

    If `stem_prefix` is given, additionally require the filename (stem) to
    start with that prefix.
    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    seed_to_path: dict[int, Path] = {}
    for path in sorted(folder.glob("*.json")):
        m = SEED_SUFFIX_RE.search(path.stem)
        if not m:
            continue
        if stem_prefix is not None and not path.stem.startswith(stem_prefix):
            continue
        seed = int(m.group(1))
        if seed in seed_to_path:
            raise RuntimeError(
                f"Two JSONs claim seed {seed}: {seed_to_path[seed]} and {path}. "
                "Pass --stem-prefix to disambiguate."
            )
        seed_to_path[seed] = path
    if not seed_to_path:
        raise FileNotFoundError(
            f"No `*_gen_s<seed>.json` files found in {folder}"
            + (f" matching prefix '{stem_prefix}'" if stem_prefix else "")
        )
    return seed_to_path


def derive_common_stem(seed_to_path: dict[int, Path]) -> str:
    """Strip the `_gen_s<seed>` suffix from the stem (must be identical across seeds)."""
    stems = {SEED_SUFFIX_RE.sub("", p.stem) for p in seed_to_path.values()}
    if len(stems) != 1:
        raise RuntimeError(
            f"Per-seed JSON stems disagree before the `_gen_s<seed>` suffix: {stems}. "
            "Pass --stem-prefix to disambiguate."
        )
    return stems.pop()


def average_per_subject_acc(
    seed_jsons: dict[int, dict],
) -> tuple[dict[int, float], dict[int, int]]:
    """Average each subject's per-seed accuracy. Returns (mean_acc, n_seeds_per_subject)."""
    per_subject_accs: dict[int, list[float]] = defaultdict(list)
    for seed, payload in seed_jsons.items():
        for sid_str, acc in payload["per_subject_acc"].items():
            per_subject_accs[int(sid_str)].append(float(acc))
    mean_acc = {sid: float(np.mean(accs)) for sid, accs in per_subject_accs.items()}
    n_seeds = {sid: len(accs) for sid, accs in per_subject_accs.items()}
    return mean_acc, n_seeds


def load_seed_npz(json_path: Path) -> dict[str, np.ndarray]:
    """Load the NPZ sibling of a per-seed JSON."""
    npz_path = json_path.with_suffix(".npz")
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found alongside {json_path}: expected {npz_path}")
    return dict(np.load(npz_path))


def main() -> None:
    args = parse_args()
    folder = args.results_root / f"{args.approach}_{args.task}_generalization"
    seed_to_path = find_per_seed_jsons(folder, args.stem_prefix)
    common_stem = derive_common_stem(seed_to_path)

    print(f"\n=== Averaging across {len(seed_to_path)} seeds ===")
    print(f"  Folder:       {folder}")
    print(f"  Common stem:  {common_stem}")
    for seed, path in sorted(seed_to_path.items()):
        print(f"  seed {seed:>4}: {path.name}")

    # Load all per-seed JSONs and NPZs.
    seed_jsons = {seed: json.loads(path.read_text()) for seed, path in seed_to_path.items()}
    seed_npzs = {seed: load_seed_npz(path) for seed, path in seed_to_path.items()}

    # Average per-subject accuracies.
    per_subject_acc, n_seeds_per_subject = average_per_subject_acc(seed_jsons)
    accs_arr = np.array(list(per_subject_acc.values()), dtype=float)
    n_subjects = len(accs_arr)
    if n_subjects == 0:
        raise RuntimeError("No subjects after averaging — input JSONs are empty?")

    # Pool per-window predictions across seeds for the macro report.
    y_true = np.concatenate([npz["y_true"] for npz in seed_npzs.values()]).astype(np.int32)
    y_pred = np.concatenate([npz["y_pred"] for npz in seed_npzs.values()]).astype(np.int32)
    y_prob = np.concatenate([npz["y_prob"] for npz in seed_npzs.values()], axis=0).astype(np.float32)
    subj_ids = np.concatenate([npz["subj_ids"] for npz in seed_npzs.values()]).astype(np.int32)
    stim_ids = np.concatenate([npz["stim_ids"] for npz in seed_npzs.values()]).astype(np.int32)

    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )

    auroc_per_class: list[float] = []
    for i in range(y_prob.shape[1]):
        y_bin = (y_true == labels[i]).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            auroc_per_class.append(float("nan"))
        else:
            auroc_per_class.append(float(roc_auc_score(y_bin, y_prob[:, i])))
    try:
        if y_prob.shape[1] == 2:
            macro_auroc = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            macro_auroc = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=labels)
            )
    except ValueError:
        macro_auroc = float("nan")

    # Build combined summary.
    per_subject_support: dict[int, int] = defaultdict(int)
    for npz in seed_npzs.values():
        for sid in np.unique(npz["subj_ids"]).tolist():
            per_subject_support[int(sid)] += int((npz["subj_ids"] == sid).sum())

    summary: dict = {
        "approach": args.approach,
        "task": args.task,
        "dataset": seed_jsons[next(iter(seed_jsons))].get("dataset"),
        "run_stem": common_stem + "_gen_avg",
        "completed_at": datetime.datetime.now().isoformat(),
        "gen_seeds": sorted(seed_jsons.keys()),
        "n_seeds": len(seed_jsons),
        "n_subjects": n_subjects,
        "subject_wise": {
            "mean_acc": float(accs_arr.mean()),
            "std_acc": float(accs_arr.std(ddof=1)) if n_subjects > 1 else float("nan"),
            "min_acc": float(accs_arr.min()),
            "max_acc": float(accs_arr.max()),
        },
        "window_wise_acc": float((y_pred == y_true).mean()),
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
        "per_subject_acc": {str(sid): acc for sid, acc in per_subject_acc.items()},
        "per_subject_support": {str(sid): n for sid, n in per_subject_support.items()},
        "n_seeds_per_subject": {str(sid): n for sid, n in n_seeds_per_subject.items()},
    }

    out_stem = args.out_name or f"{common_stem}_gen_avg"
    json_path = folder / f"{out_stem}.json"
    npz_path = folder / f"{out_stem}.npz"
    json_path.write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        npz_path,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        subj_ids=subj_ids,
        stim_ids=stim_ids,
        labels=np.array(labels, dtype=np.int32),
    )

    print(f"\nN subjects: {n_subjects}  |  seeds per subject: "
          f"min={min(n_seeds_per_subject.values())} max={max(n_seeds_per_subject.values())}")
    print(f"Subject-wise: mean={accs_arr.mean():.4f}  std={accs_arr.std(ddof=1):.4f}")
    print(f"\nSaved → {json_path}")
    print(f"Saved → {npz_path}")


if __name__ == "__main__":
    main()
