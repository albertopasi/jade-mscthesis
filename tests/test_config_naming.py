"""Tests for cfg.run_name_stem() and cfg.run_name() across LP/SFT/JADE.

These functions are the canonical naming convention for every artifact a
run produces (checkpoint dir, W&B run name, main-results JSON stem). If a
new HP is added but not encoded in run_name_stem, two different sweep
cells can collide on disk and silently overwrite each other.

These tests pin the convention so a careless edit to one of the config
files becomes a test failure rather than a data loss bug.
"""

from __future__ import annotations

import pytest

from src.approaches.fine_tuning.config import FTConfig
from src.approaches.jade.config import JADEConfig
from src.approaches.linear_probing.config import LPConfig

# ── JADEConfig ────────────────────────────────────────────────────────────


class TestJADENaming:
    def test_run_name_starts_with_jade_prefix(self):
        cfg = JADEConfig()
        assert cfg.run_name_stem().startswith("jade_")

    def test_run_name_is_stem_plus_fold(self):
        cfg = JADEConfig()
        assert cfg.run_name(1) == f"{cfg.run_name_stem()}_fold_1"
        assert cfg.run_name(10) == f"{cfg.run_name_stem()}_fold_10"

    def test_gen_seed_embeds_into_stem(self):
        cfg = JADEConfig()
        without = cfg.run_name_stem(None)
        with_seed = cfg.run_name_stem(gen_seed=123)
        assert "gen_s123" in with_seed
        assert "gen" not in without

    def test_gen_seed_propagates_to_run_name(self):
        cfg = JADEConfig()
        rn = cfg.run_name(fold_idx=3, gen_seed=789)
        assert "gen_s789" in rn
        assert rn.endswith("_fold_3")

    def test_thesis_9class_winner_exact_match(self):
        """Pin the exact thesis-final 9-class JADE stem against drift."""
        cfg = JADEConfig(
            dataset="faced",
            task_mode="9-class",
            window_size=2000,
            stride=2000,
            pooling="no",
            full_ft=True,
            batch_size=256,
            ft_lr=4e-4,
            lora_rank=16,
            supcon_alpha=0.3,
            supcon_temperature=0.2,
            supcon_repr="context",
        )
        assert cfg.run_name_stem() == (
            "jade_faced_9-class_w10s10_pool_no_r16_a0.3_t0.2_context_b256_lr0.0004_fullft"
        )

    def test_thesis_binary_winner_exact_match(self):
        cfg = JADEConfig(
            dataset="faced",
            task_mode="binary",
            window_size=2000,
            stride=2000,
            pooling="no",
            full_ft=True,
            batch_size=128,
            ft_lr=1e-4,
            lora_rank=16,
            supcon_alpha=0.2,
            supcon_temperature=0.05,
            supcon_repr="context",
        )
        assert cfg.run_name_stem() == (
            "jade_faced_binary_w10s10_pool_no_r16_a0.2_t0.05_context_b128_lr0.0001_fullft"
        )

    @pytest.mark.parametrize(
        "kwarg,new_value",
        [
            ("supcon_alpha", 0.3),  # default 0.5
            ("supcon_temperature", 0.2),  # default 0.1
            ("batch_size", 512),
            ("ft_lr", 2e-4),
            ("lora_rank", 8),
            ("supcon_repr", "mean"),
            ("pooling", "last"),
        ],
    )
    def test_each_hp_distinguishes_stem(self, kwarg, new_value):
        """Every sweep-distinguishing HP must change the run_name_stem.

        This is the regression test for "I added a new HP and forgot to encode it
        in run_name_stem, so two sweep cells collided on disk and overwrote each
        other".
        """
        base = JADEConfig()
        # Sanity: don't pick the same value as the default.
        assert getattr(base, kwarg) != new_value, f"Test bug: new_value matches default for {kwarg}"
        variant = JADEConfig(**{kwarg: new_value})
        assert base.run_name_stem() != variant.run_name_stem(), (
            f"Changing {kwarg} from {getattr(base, kwarg)} to {new_value} did not change run_name_stem"
        )


# ── FTConfig ──────────────────────────────────────────────────────────────


class TestFTNaming:
    def test_prefix(self):
        cfg = FTConfig()
        assert cfg.run_name_stem().startswith("ft_")

    def test_stem_plus_fold(self):
        cfg = FTConfig()
        assert cfg.run_name(5) == f"{cfg.run_name_stem()}_fold_5"

    def test_gen_seed_embedded(self):
        cfg = FTConfig()
        with_seed = cfg.run_name_stem(gen_seed=456)
        assert "gen_s456" in with_seed

    def test_nomixup_tag(self):
        cfg = FTConfig(use_mixup=False)
        assert "_nomixup" in cfg.run_name_stem()
        cfg2 = FTConfig(use_mixup=True)
        assert "_nomixup" not in cfg2.run_name_stem()

    def test_fullft_tag(self):
        cfg = FTConfig(full_ft=True)
        assert "_fullft" in cfg.run_name_stem()

    def test_thesis_9class_winner_exact_match(self):
        cfg = FTConfig(
            dataset="faced",
            task_mode="9-class",
            window_size=2000,
            stride=2000,
            pooling="no",
            full_ft=True,
            use_mixup=False,
            batch_size=256,
            ft_lr=4e-4,
            lora_rank=16,
        )
        assert cfg.run_name_stem() == (
            "ft_faced_9-class_w10s10_pool_no_r16_b256_lr0.0004_nomixup_fullft"
        )

    @pytest.mark.parametrize(
        "kwarg,new_value",
        [
            ("batch_size", 64),
            ("ft_lr", 2e-4),
            ("lora_rank", 8),
            ("pooling", "last"),
            ("use_mixup", True),
            ("full_ft", False),
        ],
    )
    def test_each_hp_distinguishes_stem(self, kwarg, new_value):
        base = FTConfig(full_ft=True, use_mixup=False, batch_size=256, ft_lr=4e-4)
        variant_kwargs = {
            "full_ft": True,
            "use_mixup": False,
            "batch_size": 256,
            "ft_lr": 4e-4,
            kwarg: new_value,
        }
        variant = FTConfig(**variant_kwargs)
        assert base.run_name_stem() != variant.run_name_stem(), (
            f"Changing {kwarg} did not change run_name_stem"
        )


# ── LPConfig ──────────────────────────────────────────────────────────────


class TestLPNaming:
    def test_prefix(self):
        cfg = LPConfig()
        assert cfg.run_name_stem().startswith("lp_")

    def test_stem_plus_fold(self):
        cfg = LPConfig()
        assert cfg.run_name(7) == f"{cfg.run_name_stem()}_fold_7"

    def test_gen_seed_embedded(self):
        cfg = LPConfig()
        assert "gen_s321" in cfg.run_name_stem(gen_seed=321)

    def test_nomixup_tag(self):
        cfg = LPConfig(use_mixup=False)
        assert "_nomixup" in cfg.run_name_stem()
        cfg2 = LPConfig(use_mixup=True)
        assert "_nomixup" not in cfg2.run_name_stem()

    def test_thesis_9class_lp_exact_match(self):
        cfg = LPConfig(
            dataset="faced",
            task_mode="9-class",
            window_size=2000,
            stride=2000,
            pooling="no",
            official_mode=True,
            use_mixup=False,
        )
        assert cfg.run_name_stem() == "lp_faced_v2_9-class_w10s10_pool_no_official_nomixup"

    @pytest.mark.parametrize(
        "kwarg,new_value",
        [
            ("pooling", "last"),
            ("use_mixup", True),
            ("task_mode", "9-class"),  # default is "binary"
            ("normalize_features", True),
        ],
    )
    def test_each_hp_distinguishes_stem(self, kwarg, new_value):
        base = LPConfig(official_mode=True, use_mixup=False)
        # Sanity: don't pick the same value as the default.
        assert getattr(base, kwarg) != new_value, f"Test bug: new_value matches default for {kwarg}"
        variant_kwargs = {"official_mode": True, "use_mixup": False, kwarg: new_value}
        variant = LPConfig(**variant_kwargs)
        assert base.run_name_stem() != variant.run_name_stem(), (
            f"Changing {kwarg} did not change run_name_stem"
        )


# ── Cross-approach invariants ─────────────────────────────────────────────


class TestCrossApproachInvariants:
    def test_stems_dont_collide_across_approaches(self):
        """LP/SFT/JADE stems must never accidentally match.

        Without the prefix difference, e.g. LP and SFT configs that share
        every other HP would write to the same JSON.
        """
        lp = LPConfig().run_name_stem()
        ft = FTConfig().run_name_stem()
        jade = JADEConfig().run_name_stem()
        assert lp != ft != jade
        assert lp != jade

    def test_no_double_underscore_or_trailing_underscore(self):
        # Stylistic — these usually indicate a tag was empty when it should not be.
        for cfg in (LPConfig(), FTConfig(), JADEConfig()):
            stem = cfg.run_name_stem()
            assert "__" not in stem, f"Double underscore in {stem}"
            assert not stem.endswith("_"), f"Trailing underscore in {stem}"

    def test_fold_suffix_is_one_indexed(self):
        # All approaches share the convention that folds are 1..N (not 0..N-1).
        # If anyone switches to 0-indexed, this breaks.
        for cfg in (LPConfig(), FTConfig(), JADEConfig()):
            assert cfg.run_name(1).endswith("_fold_1")
            assert cfg.run_name(10).endswith("_fold_10")
