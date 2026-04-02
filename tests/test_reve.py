"""
Tests for REVE model loading, position bank, and shared model utilities.

Covers:
  - get_channel_names: correct channels for FACED and THU-EP
  - load_reve_and_positions: model loads, eval mode, frozen params, output shapes
  - Position bank: electrode lookup, known channel resolution
  - RMSNorm: mathematical correctness
  - compute_n_patches: window → patch count formula
  - Shared config constants: paths, device, dataset defaults
  - REVE forward pass: output shape for known input
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# get_channel_names


class TestGetChannelNames:
    """Tests for get_channel_names."""

    def test_faced_32_channels(self):
        """FACED returns 32 electrode names."""
        from src.approaches.shared.reve import get_channel_names

        names = get_channel_names("faced")
        assert len(names) == 32

    def test_thuep_30_channels(self):
        """THU-EP returns 30 electrode names (A1, A2 removed)."""
        from src.approaches.shared.reve import get_channel_names

        names = get_channel_names("thu-ep")
        assert len(names) == 30
        assert "A1" not in names
        assert "A2" not in names

    def test_faced_contains_a1_a2(self):
        """FACED channel list includes A1 and A2 (not removed for FACED)."""
        from src.approaches.shared.reve import get_channel_names

        names = get_channel_names("faced")
        assert "A1" in names
        assert "A2" in names

    def test_channels_are_strings(self):
        """All channel names should be strings."""
        from src.approaches.shared.reve import get_channel_names

        for dataset in ["faced", "thu-ep"]:
            names = get_channel_names(dataset)
            assert all(isinstance(n, str) for n in names)

    def test_no_duplicates(self):
        """Channel names should not contain duplicates."""
        from src.approaches.shared.reve import get_channel_names

        for dataset in ["faced", "thu-ep"]:
            names = get_channel_names(dataset)
            assert len(names) == len(set(names)), f"Duplicate channels in {dataset}"


# load_reve_and_positions


class TestLoadReveAndPositions:
    """Tests for loading the REVE model and position bank.

    These tests require the actual pretrained model files to be present.
    They are marked with @pytest.mark.reve to allow selective execution.
    """

    @pytest.fixture(autouse=True)
    def _check_model_files(self):
        """Skip if model files are not available."""
        from src.approaches.shared.config import REVE_MODEL_PATH, REVE_POS_PATH

        if not REVE_MODEL_PATH.exists():
            pytest.skip(f"REVE model not found at {REVE_MODEL_PATH}")
        if not REVE_POS_PATH.exists():
            pytest.skip(f"REVE positions not found at {REVE_POS_PATH}")

    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.fixture
    def faced_model_and_pos(self, device):
        """Load REVE + positions for FACED (cached per test session)."""
        from src.approaches.shared.reve import get_channel_names, load_reve_and_positions

        names = get_channel_names("faced")
        return load_reve_and_positions(names, device=device)

    def test_model_returns_module(self, faced_model_and_pos):
        """load_reve_and_positions returns an nn.Module."""
        model, _ = faced_model_and_pos
        assert isinstance(model, torch.nn.Module)

    def test_model_in_eval_mode(self, faced_model_and_pos):
        """Loaded model should be in eval mode."""
        model, _ = faced_model_and_pos
        assert not model.training

    def test_model_params_frozen(self, faced_model_and_pos):
        """All model parameters should have requires_grad=False."""
        model, _ = faced_model_and_pos
        for name, param in model.named_parameters():
            assert not param.requires_grad, f"Parameter {name} is not frozen"

    def test_position_tensor_shape_faced(self, faced_model_and_pos):
        """Position tensor for FACED: (32, 3) — 32 channels, xyz coords."""
        _, pos = faced_model_and_pos
        assert pos.shape == (32, 3)

    def test_position_tensor_shape_thuep(self, device):
        """Position tensor for THU-EP: (30, 3) — 30 channels, xyz coords."""
        from src.approaches.shared.reve import get_channel_names, load_reve_and_positions

        names = get_channel_names("thu-ep")
        _, pos = load_reve_and_positions(names, device=device)
        assert pos.shape == (30, 3)

    def test_position_tensor_dtype(self, faced_model_and_pos):
        """Position tensor should be a float type."""
        _, pos = faced_model_and_pos
        assert pos.dtype in (torch.float32, torch.float16, torch.bfloat16)

    def test_position_tensor_finite(self, faced_model_and_pos):
        """Position values should be finite (no NaN/Inf)."""
        _, pos = faced_model_and_pos
        assert torch.isfinite(pos).all()

    def test_forward_pass_output_shape(self, faced_model_and_pos, device):
        """REVE forward: (B, 32, 2000) → (B, 32, n_patches, 512)."""
        from src.approaches.shared.model_utils import compute_n_patches

        model, pos = faced_model_and_pos
        batch_size = 2
        n_channels = 32
        window_size = 2000

        eeg = torch.randn(batch_size, n_channels, window_size, device=device)
        pos_batch = pos.unsqueeze(0).expand(batch_size, -1, -1)

        with torch.no_grad():
            out = model(eeg, pos_batch)

        n_patches = compute_n_patches(window_size)
        assert out.shape == (batch_size, n_channels, n_patches, 512)

    def test_forward_pass_deterministic(self, faced_model_and_pos, device):
        """Two forward passes with same input should produce identical output."""
        model, pos = faced_model_and_pos
        eeg = torch.randn(1, 32, 2000, device=device)
        pos_batch = pos.unsqueeze(0)

        with torch.no_grad():
            out1 = model(eeg, pos_batch)
            out2 = model(eeg, pos_batch)

        torch.testing.assert_close(out1, out2)


# RMSNorm


class TestRMSNorm:
    """Tests for the RMSNorm building block."""

    def test_output_shape(self):
        """RMSNorm preserves input shape."""
        from src.approaches.shared.model_utils import RMSNorm

        norm = RMSNorm(dim=512)
        x = torch.randn(4, 10, 512)
        out = norm(x)
        assert out.shape == x.shape

    def test_mathematical_correctness(self):
        """Verify RMSNorm formula: y = (x / rms) * scale, where rms = sqrt(mean(x^2) + eps)."""
        from src.approaches.shared.model_utils import RMSNorm

        dim = 8
        norm = RMSNorm(dim=dim, eps=1e-8)
        # Set scale to ones (default) for easier verification
        with torch.no_grad():
            norm.scale.fill_(1.0)

        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
        out = norm(x)

        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-8)
        expected = x / rms
        torch.testing.assert_close(out, expected)

    def test_zero_input(self):
        """RMSNorm handles zero input without NaN (due to eps)."""
        from src.approaches.shared.model_utils import RMSNorm

        norm = RMSNorm(dim=16)
        x = torch.zeros(2, 16)
        out = norm(x)
        assert torch.isfinite(out).all()

    def test_learnable_scale(self):
        """The scale parameter should be learnable (requires_grad=True)."""
        from src.approaches.shared.model_utils import RMSNorm

        norm = RMSNorm(dim=64)
        assert norm.scale.requires_grad


# compute_n_patches


class TestComputeNPatches:
    """Tests for compute_n_patches."""

    def test_window_2000(self):
        """window_size=2000, patch=200, overlap=20 → 10 patches.

        step = 200 - 20 = 180
        n_patches = (2000 - 200) / 180 + 1 = 1800/180 + 1 = 10 + 1 = 11

        Wait — let's recalculate:
        (2000 - 200) // 180 + 1 = 1800 // 180 + 1 = 10 + 1 = 11
        """
        from src.approaches.shared.model_utils import compute_n_patches

        assert compute_n_patches(2000) == 11

    def test_window_6000(self):
        """Full 30s recording: window_size=6000 → 33 patches.

        (6000 - 200) // 180 + 1 = 5800 // 180 + 1 = 32 + 1 = 33
        """
        from src.approaches.shared.model_utils import compute_n_patches

        assert compute_n_patches(6000) == 33

    def test_window_equals_patch(self):
        """When window == patch_size, exactly 1 patch."""
        from src.approaches.shared.model_utils import compute_n_patches

        assert compute_n_patches(200) == 1

    def test_custom_parameters(self):
        """Custom patch_size and overlap."""
        from src.approaches.shared.model_utils import compute_n_patches

        # patch=100, overlap=10, step=90
        # window=1000: (1000-100)//90+1 = 900//90+1 = 10+1 = 11
        assert compute_n_patches(1000, patch_size=100, overlap=10) == 11


# Shared config constants


class TestSharedConfig:
    """Tests for shared config constants."""

    def test_project_root_exists(self):
        """PROJECT_ROOT should point to an existing directory."""
        from src.approaches.shared.config import PROJECT_ROOT

        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.is_dir()

    def test_data_roots_keys(self):
        """DATA_ROOTS has entries for both datasets."""
        from src.approaches.shared.config import DATA_ROOTS

        assert "faced" in DATA_ROOTS
        assert "thu-ep" in DATA_ROOTS

    def test_reve_paths_exist(self):
        """REVE model and position paths should exist."""
        from src.approaches.shared.config import REVE_MODEL_PATH, REVE_POS_PATH

        assert REVE_MODEL_PATH.exists(), f"REVE model not found: {REVE_MODEL_PATH}"
        assert REVE_POS_PATH.exists(), f"REVE positions not found: {REVE_POS_PATH}"

    def test_sampling_rate(self):
        """Sampling rate is 200 Hz (post-preprocessing)."""
        from src.approaches.shared.config import SAMPLING_RATE

        assert SAMPLING_RATE == 200

    def test_dataset_defaults(self):
        """Dataset defaults have correct channel counts and scale factors."""
        from src.approaches.shared.config import DATASET_DEFAULTS

        assert DATASET_DEFAULTS["faced"]["n_channels"] == 32
        assert DATASET_DEFAULTS["thu-ep"]["n_channels"] == 30
        assert DATASET_DEFAULTS["faced"]["scale_factor"] == 1000.0
        assert DATASET_DEFAULTS["thu-ep"]["scale_factor"] == 1000.0

    def test_device_is_valid(self):
        """DEVICE should be 'cuda' or 'cpu'."""
        from src.approaches.shared.config import DEVICE

        assert DEVICE in ("cuda", "cpu")
