"""
Visualize FACED preprocessed EEG data.

Loads a FACED preprocessed .npy file (28, 32, 6000) and displays
all 32 channels as EEG waveforms.

Usage:
    uv run python -m src.exploration.faced.visualize_preprocessed
    uv run python -m src.exploration.faced.visualize_preprocessed --subject 5
    uv run python -m src.exploration.faced.visualize_preprocessed --sample 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "FACED" / "preprocessed_v2"

# FACED preprocessed specs
N_SAMPLES = 28
N_CHANNELS = 32
N_TIMEPOINTS = 6000
TARGET_SFREQ = 200
DURATION_S = N_TIMEPOINTS / TARGET_SFREQ  # 30 seconds


def load_subject(subject_id: int) -> np.ndarray:
    """Load preprocessed FACED data for a subject."""
    npy_file = DATA_DIR / f"sub{subject_id:03d}.npy"
    if not npy_file.exists():
        raise FileNotFoundError(f"Subject file not found: {npy_file}")
    
    data = np.load(npy_file)  # (28, 32, 6000)
    print(f"Loaded {npy_file.name}: shape={data.shape}, dtype={data.dtype}")
    return data


def visualize_sample(data: np.ndarray, sample_idx: int = 0) -> None:
    """Visualize a single sample with all 32 channels."""
    if sample_idx >= data.shape[0]:
        raise ValueError(f"Sample {sample_idx} out of range [0, {data.shape[0]-1}]")
    
    sample = data[sample_idx]  # (32, 6000)
    
    # Create time axis in seconds
    time = np.arange(N_TIMEPOINTS) / TARGET_SFREQ
    
    # Create figure with subplots for each channel
    fig, axes = plt.subplots(N_CHANNELS, 1, figsize=(14, 20))
    fig.suptitle(f"FACED EEG - All {N_CHANNELS} Channels (Sample {sample_idx})", fontsize=14, fontweight="bold")
    
    for ch_idx, ax in enumerate(axes):
        ax.plot(time, sample[ch_idx], linewidth=0.5, color="steelblue")
        ax.set_ylabel(f"Ch {ch_idx}", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([sample.min() - 1, sample.max() + 1])  # Common scale
        
        if ch_idx < N_CHANNELS - 1:
            ax.set_xticklabels([])
    
    axes[-1].set_xlabel("Time (seconds)", fontsize=10)
    
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize FACED preprocessed EEG data")
    parser.add_argument("--subject", "-s", type=int, default=0,
                        help="Subject ID (0-indexed, default: 0)")
    parser.add_argument("--sample", "-n", type=int, default=0,
                        help="Sample index within subject (0-27, default: 0)")
    args = parser.parse_args()
    
    try:
        data = load_subject(args.subject)
        print(f"Visualizing sample {args.sample} of subject {args.subject} ({args.sample}/{data.shape[0]-1})")
        visualize_sample(data, args.sample)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
