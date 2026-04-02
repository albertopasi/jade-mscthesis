"""
Visualize electrode positions for FACED or THU EP datasets.

Loads electrode names from dataset configurations, fetches their 3D positions
from REVE position bank, and visualizes them in 3D and 2D (topographic) views.

Features:
- Choose between FACED and THU EP datasets
- Loads electrode names from configs
- Visualizes all electrodes in 3D space and topomap
- Shows electrode labels for easy identification
- Handles reference channel removal (THU EP: A1, A2)

Usage:
    # Interactive mode (prompts for dataset choice)
    uv run python -m src.exploration.visualize_electrode_positions

    # Direct selection
    uv run python -m src.exploration.visualize_electrode_positions --dataset faced
    uv run python -m src.exploration.visualize_electrode_positions --dataset thu_ep

    # Show processing channels (default) or all channels
    uv run python -m src.exploration.visualize_electrode_positions --dataset thu_ep --show-all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# File paths
REVE_POSITIONS_FILE = (
    PROJECT_ROOT / "src" / "exploration" / "electrodes_pos" / "reve_all_positions.json"
)
THU_EP_CONFIG_FILE = PROJECT_ROOT / "configs" / "thu_ep.yml"

# THU EP electrode names (standard 10-20 extended)
THU_EP_ALL_CHANNELS = [
    "Fp1",
    "Fp2",
    "Fz",
    "F3",
    "F4",
    "F7",
    "F8",
    "FC1",
    "FC2",
    "FC5",
    "FC6",
    "Cz",
    "C3",
    "C4",
    "T7",
    "T8",
    "A1",
    "A2",
    "CP1",
    "CP2",
    "CP5",
    "CP6",
    "Pz",
    "P3",
    "P4",
    "P7",
    "P8",
    "PO3",
    "PO4",
    "Oz",
    "O1",
    "O2",
]

# THU EP channels to remove (mastoid references)
THU_EP_REMOVE_CHANNELS = ["A1", "A2"]

# FACED
FACED_CHANNELS = [
    "FP1",
    "FP2",
    "FZ",
    "F3",
    "F4",
    "F7",
    "F8",
    "FC1",
    "FC2",
    "FC5",
    "FC6",
    "CZ",
    "C3",
    "C4",
    "T3",
    "T4",
    "CP1",
    "CP2",
    "CP5",
    "CP6",
    "PZ",
    "P3",
    "P4",
    "T5",
    "T6",
    "PO3",
    "PO4",
    "OZ",
    "O1",
    "O2",
    "A2",
    "A1",
]


def load_reve_positions(reve_file: Path) -> dict:
    """Load REVE position bank (3D electrode coordinates)."""
    if not reve_file.exists():
        raise FileNotFoundError(f"REVE positions file not found: {reve_file}")

    print(f"✓ Loading REVE positions from: {reve_file.name}")
    with open(reve_file, "r") as f:
        positions = json.load(f)

    print(f"  Found {len(positions)} electrode positions in REVE database")
    return positions


def load_thu_ep_config(config_file: Path) -> dict:
    """Load THU EP configuration."""
    if not config_file.exists():
        raise FileNotFoundError(f"THU EP config not found: {config_file}")

    print(f"✓ Loading THU EP config from: {config_file.name}")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_dataset_electrodes(dataset: str, show_all: bool = False) -> tuple[list[str], str]:
    """
    Get electrode names for the chosen dataset.

    Returns:
        (electrode_names, dataset_label)
    """
    if dataset.lower() == "faced":
        channels = FACED_CHANNELS
        label = "FACED (32 channels)"
    elif dataset.lower() == "thu_ep":
        channels = THU_EP_ALL_CHANNELS
        if not show_all:
            channels = [ch for ch in channels if ch not in THU_EP_REMOVE_CHANNELS]
            label = "THU EP (30 channels, A1/A2 removed)"
        else:
            label = "THU EP (32 channels, all)"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return channels, label


def fetch_electrode_positions(electrode_names: list[str], reve_positions: dict) -> dict:
    """
    Fetch 3D positions for given electrode names from REVE database.

    Returns dict mapping electrode names to (x, y, z) coordinates.
    """
    positions = {}
    missing = []

    for name in electrode_names:
        if name in reve_positions:
            pos = reve_positions[name]
            positions[name] = np.array([pos["x"], pos["y"], pos["z"]])
        else:
            missing.append(name)

    if missing:
        print(f"⚠ Missing {len(missing)} electrodes in REVE database: {missing}")

    print(f"✓ Found {len(positions)}/{len(electrode_names)} electrode positions")
    return positions


def visualize_electrodes_3d_views(
    electrode_names: list[str], positions_dict: dict, dataset_label: str
) -> None:
    """
    Create 3D and 2D visualizations of electrode positions.

    Uses MNE's visualization functions to create:
    - 3D scatter plot with labels
    - 2D topomap
    """
    if not positions_dict:
        print("No positions to visualize!")
        return

    print(f"\nVisualizing {len(positions_dict)} electrodes for: {dataset_label}")
    print(f"{'=' * 70}")

    # Create MNE info object
    ch_names = list(positions_dict.keys())
    info = mne.create_info(ch_names, sfreq=500, ch_types="eeg")

    # Create dummy raw data
    raw = mne.io.RawArray(np.zeros((len(ch_names), 1000)), info)

    # Create montage from positions
    montage = mne.channels.make_dig_montage(ch_pos=positions_dict, coord_frame="head")
    raw.set_montage(montage)

    # Plot 3D view
    print("\n📊 Rendering 3D view...")
    fig_3d = mne.viz.plot_sensors(
        raw.info,
        kind="3d",
        title=f"{dataset_label} - 3D Electrode Positions",
        show_names=True,
        sphere=(0, 0, 0, 0.1),
    )

    # Plot 2D topomap view
    print("📊 Rendering 2D topomap view...")
    fig_2d = mne.viz.plot_sensors(
        raw.info,
        kind="topomap",
        title=f"{dataset_label} - Topomap (Top-Down View)",
        show_names=True,
    )

    print("\n✓ Visualizations ready. Close figures to continue.")
    plt.show()


def visualize_custom_3d(
    electrode_names: list[str], positions_dict: dict, dataset_label: str
) -> None:
    """
    Create custom 3D visualization with multiple viewpoints.

    Shows:
    - 3D scatter plot from multiple angles
    - Labels for all electrodes
    """
    if not positions_dict:
        print("No positions to visualize!")
        return

    print(f"\nCreating custom 3D visualization for: {dataset_label}")
    print(f"{'=' * 70}")

    # Extract coordinates
    positions_array = np.array([positions_dict[name] for name in electrode_names])
    x, y, z = positions_array[:, 0], positions_array[:, 1], positions_array[:, 2]

    # Create figure with multiple subplots (different views)
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"{dataset_label} - Multiple 3D Views", fontsize=14, fontweight="bold")

    views = [
        ("Top-Down (Z)", (90, 0)),
        ("Front (Y)", (0, 0)),
        ("Side (X)", (0, 90)),
        ("Isometric", (45, 45)),
    ]

    for idx, (view_name, (elev, azim)) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")

        # Plot electrodes as scatter
        ax.scatter(x, y, z, s=100, c="steelblue", marker="o", edgecolors="black", linewidth=0.5)

        # Add labels
        for name, (xi, yi, zi) in zip(electrode_names, positions_array):
            ax.text(xi, yi, zi, name, fontsize=7, ha="center", va="center")

        # Set labels and title
        ax.set_xlabel("X", fontsize=9)
        ax.set_ylabel("Y", fontsize=9)
        ax.set_zlabel("Z", fontsize=9)
        ax.set_title(f"{view_name}", fontsize=11, fontweight="bold")

        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)

        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])

        # Set limits to approximately [-0.1, 0.1]
        ax.set_xlim([-0.12, 0.12])
        ax.set_ylim([-0.12, 0.12])
        ax.set_zlim([-0.12, 0.12])

        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize electrode positions for FACED or THU EP datasets"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        choices=["faced", "thu_ep"],
        help="Dataset to visualize (faced or thu_ep). If not provided, prompts interactively.",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all channels including reference channels (THU EP only). Default: removes A1/A2.",
    )
    parser.add_argument(
        "--mne-only",
        action="store_true",
        help="Use only MNE visualization (default: shows both MNE + custom multi-view).",
    )
    args = parser.parse_args()

    # Determine dataset
    dataset = args.dataset
    if not dataset:
        print("\nAvailable datasets:")
        print("  1. faced    - FACED dataset (32 channels)")
        print("  2. thu_ep   - THU EP dataset (32 channels, removes A1/A2 for preprocessing)")
        choice = input("\nSelect dataset (1 or 2, or name): ").strip().lower()
        if choice in ["1", "faced"]:
            dataset = "faced"
        elif choice in ["2", "thu_ep"]:
            dataset = "thu_ep"
        else:
            print("Invalid choice. Using FACED.")
            dataset = "faced"

    try:
        # Load REVE positions
        print("\n" + "=" * 70)
        print("ELECTRODE POSITION VISUALIZATION")
        print("=" * 70)
        reve_positions = load_reve_positions(REVE_POSITIONS_FILE)

        # Get dataset electrodes
        print(f"\nLoading {dataset.upper()} electrode configuration...")
        electrode_names, dataset_label = get_dataset_electrodes(dataset, args.show_all)
        print(f"✓ Dataset: {dataset_label}")
        print(f"  Channels: {', '.join(electrode_names)}")

        # Fetch positions for this dataset
        print("\nMatching electrodes with REVE positions...")
        positions_dict = fetch_electrode_positions(electrode_names, reve_positions)

        # Visualize
        print("\n" + "=" * 70)
        visualize_electrodes_3d_views(electrode_names, positions_dict, dataset_label)

        if not args.mne_only:
            visualize_custom_3d(electrode_names, positions_dict, dataset_label)

        print("\n✓ Visualization complete!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
