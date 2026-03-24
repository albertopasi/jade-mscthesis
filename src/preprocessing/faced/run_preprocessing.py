"""
Run FACED preprocessing pipeline.

Converts raw FACED .pkl files (28, 32, 7500) @ 250 Hz
to preprocessed .npy files (28, 32, 6000) @ 200 Hz.

Steps (matching reve_official/preprocessing/preprocessing_faced.py):
  1. Load sub{NNN}.pkl  →  numpy array (28, 32, 7500)
  2. Resample axis=2: 7500 → 6000  (250 Hz → 200 Hz)
  3. Save as sub{NNN}.npy  →  (28, 32, 6000) float32

Usage:
    # Process all 123 subjects
    uv run python -m src.preprocessing.faced.run_preprocessing

    # Process specific subjects
    uv run python -m src.preprocessing.faced.run_preprocessing --subjects 0 1 2

    # Validate preprocessed output
    uv run python -m src.preprocessing.faced.run_preprocessing --validate
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DIR   = PROJECT_ROOT / "data" / "FACED" / "Processed_data"
OUT_DIR   = PROJECT_ROOT / "data" / "FACED" / "preprocessed_v2"

N_SUBJECTS       = 123
ORIG_SFREQ       = 250
TARGET_SFREQ     = 200
ORIG_TIMEPOINTS  = 7500   # 30 s × 250 Hz
TARGET_TIMEPOINTS = 6000  # 30 s × 200 Hz
EXPECTED_SHAPE   = (28, 32, TARGET_TIMEPOINTS)


def preprocess_subject(subject_id: int) -> None:
    pkl_path = RAW_DIR / f"sub{subject_id:03d}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Raw file not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        array = pickle.load(f)  # (28, 32, 7500)

    if array.shape[2] != TARGET_TIMEPOINTS:
        array = scipy_signal.resample(array, TARGET_TIMEPOINTS, axis=2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"sub{subject_id:03d}.npy"
    np.save(out_path, array.astype(np.float32))


def validate_subject(subject_id: int) -> dict:
    npy_path = OUT_DIR / f"sub{subject_id:03d}.npy"
    if not npy_path.exists():
        return {"valid": False, "error": "file not found"}

    arr = np.load(npy_path)
    shape_valid = arr.shape == EXPECTED_SHAPE
    has_nan = bool(np.isnan(arr).any())
    has_inf = bool(np.isinf(arr).any())

    return {
        "valid":       shape_valid and not has_nan and not has_inf,
        "shape":       arr.shape,
        "shape_valid": shape_valid,
        "has_nan":     has_nan,
        "has_inf":     has_inf,
        "min":         float(arr.min()),
        "max":         float(arr.max()),
        "mean":        float(arr.mean()),
        "std":         float(arr.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FACED EEG preprocessing")
    parser.add_argument("--subjects", "-s", nargs="+", type=int,
                        help="Subject IDs to process (0-indexed). Processes all if omitted.")
    parser.add_argument("--validate", "-v", action="store_true",
                        help="Validate preprocessed files instead of processing.")
    args = parser.parse_args()

    subject_ids = args.subjects if args.subjects else list(range(N_SUBJECTS))

    if args.validate:
        print(f"Validating {len(subject_ids)} subject(s) in {OUT_DIR} …")
        all_valid = True
        for sid in subject_ids:
            v = validate_subject(sid)
            status = "✓" if v["valid"] else "✗"
            if v.get("error"):
                print(f"  {status} sub{sid:03d}: {v['error']}")
            else:
                print(
                    f"  {status} sub{sid:03d}: shape={v['shape']}  "
                    f"range=[{v['min']:.1f}, {v['max']:.1f}]  "
                    f"nan={v['has_nan']}  inf={v['has_inf']}"
                )
            if not v["valid"]:
                all_valid = False
        print("All valid." if all_valid else "Some subjects failed.")
        return

    print(f"Processing {len(subject_ids)} subject(s): {RAW_DIR} → {OUT_DIR}")
    t0 = time.time()
    ok, fail = 0, 0
    for sid in subject_ids:
        t_sub = time.time()
        try:
            preprocess_subject(sid)
            dt = time.time() - t_sub
            print(f"  sub{sid:03d}  done in {dt:.1f}s")
            ok += 1
        except Exception as e:
            print(f"  sub{sid:03d}  FAILED: {e}")
            fail += 1

    print(f"\nDone — {ok} succeeded, {fail} failed  ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
