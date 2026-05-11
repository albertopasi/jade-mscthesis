# Runs Inventory

All existing summary files organised by approach, dataset, task, and evaluation mode.
Missing runs (gaps in the matrix) are listed at the bottom of each section.

Legend:
- **CV** = 10-fold cross-subject CV (standard)
- **Gen** = stimulus-generalisation split (s123 / s456 / s789 + cross-seed aggregate)
- **ReveSplit** = official REVE static split (train 0-79 / val 80-99 / test 100-122)
- **B** = batch size (default 64 unless noted)
- **lr** = ft_lr (default 1e-4 unless noted)

---

## 1. Linear Probing (`lp_checkpoints`) — FACED only

Files in the root of `lp_checkpoints/` use `pool_no` (flatten) or `pool_last`.
Files in `lp_checkpoints/other/` use `nopool_flat` (flatten), `pool` (last), or `nopool_mean` (mean), with varied windows and strides.
All runs use mixup unless noted.

| Task | Pool | Window×Stride | Mixup | CV | Gen |
|------|------|---------------|-------|----|-----|
| 9-class | flat | w10s10 | yes | ✓ | ✓ |
| 9-class | flat | w10s10 | no | ✓ | — |
| 9-class | flat | w10s5 | yes | ✓ | ✓ |
| 9-class | last | w10s10 | yes | ✓ | — |
| binary | flat | w10s10 | yes | ✓ | ✓ |
| binary | flat | w10s10 | no | ✓ | — |
| binary | flat | w10s5 | yes | ✓ | ✓ |
| binary | flat | w8s4 | yes | ✓ | — |
| binary | flat | w5s5 | yes | ✓ | — |
| binary | flat | w5s2 | yes | ✓ | — |
| binary | last | w10s10 | yes | ✓ | ✓ |

> Note: `other/` gen runs include seeds s101, s123, s202, s456, s789 + cross-seed aggregate (5 seeds, vs 3 in the root folder).

**Missing:**
- [ ] 9-class last w10s10: Gen
- [ ] 9-class flat w10s10 no-mixup: Gen
- [ ] binary flat w5s2, w5s5, w8s4: Gen
- [ ] binary flat w10s10 no-mixup: Gen
- [ ] binary mean w10s5: Gen
- [ ] mean pool for w10s10 (both tasks) — only w10s5 exists for mean

---

## 2. Fine-Tuning (`ft_checkpoints`)

### 2a. FACED — FullFT (`--fullft --nomixup`)

| Task | B | lr | CV | Gen s123 | Gen s456 | Gen s789 | Gen cross-seed | ReveSplit |
|------|---|----|----|----------|----------|----------|---------------|-----------|
| 9-class | 64 | 1e-4 | ✓ | ✓ (b128 fallback) | ✓ (b128 fallback) | ✓ (b128 fallback) | ✓ | ✓ |
| 9-class | 256 | 4e-4 | ✓ | ✓ | ✓ | ✓ | — | — |
| binary | 64 | 1e-4 | ✓ | — | — | — | — | ✓ |
| binary | 64 | 1e-4 | — | ✓ (b128 lr1e-4) | ✓ (b128 lr1e-4) | ✓ (b128 lr1e-4) | — | — |
| binary | 256 | 1e-4 | ✓ | — | — | — | — | — |
| binary | 256 | 2e-4 | ✓ | — | — | — | — | — |

> Note: Gen files for 9-class CV use `b128_lr0.0001` in the name (old recipe); the main CV result uses default b64/lr1e-4.
> The `summary_faced_9-class_w10s10_pool_no_r16_fullft.json` (no `_nomixup`) is an early run with mixup on.

**Missing for FACED FullFT:**
- [ ] 9-class Gen cross-seed for B=256 lr=4e-4 (3 per-seed files exist; aggregate missing)
- [ ] binary B=256 lr=1e-4 Gen (s123/s456/s789/cross-seed)
- [ ] binary B=256 lr=2e-4 Gen (s123/s456/s789/cross-seed)
- [ ] binary Gen cross-seed aggregate (for the b128 lr1e-4 seeds that do exist)
- [ ] ReveSplit for binary B=256 lr=1e-4 or lr=2e-4 (current ReveSplit uses default b64)

### 2b. FACED — LoRA (`--nomixup`, no `--fullft`)

| Task | B | lr | CV | Gen s123 | Gen s456 | Gen s789 | Gen cross-seed |
|------|---|----|----|----------|----------|----------|---------------|
| 9-class | 64 | 1e-4 | ✓ | ✓ | ✓ | ✓ | ✓ |
| binary | 64 | 1e-4 | ✓ | ✓ | ✓ | ✓ | ✓ |

> Also exists: `summary_faced_9-class_w10s10_pool_last_r16_nomixup` (pool=last, LoRA) and `summary_faced_binary_w10s10_pool_last_r16_nomixup` — non-standard pooling, probably ablation runs.

**Missing for FACED LoRA:**
- [ ] 9-class at B=256 lr=4e-4 (winner recipe from bulletproof sweep) — CV + Gen
- [ ] binary at B=256 lr=1e-4 (winner recipe) — CV + Gen

### 2c. THU-EP — FullFT (`--fullft --nomixup`)

| Task | B | lr | CV | Gen s123 | Gen s456 | Gen s789 | Gen cross-seed |
|------|---|----|----|----------|----------|----------|---------------|
| 9-class | 64 | 1e-4 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 9-class | 256 | 4e-4 | ✓ | — | — | — | — |
| binary | 64 | 1e-4 | ✓ | ✓ | ✓ | ✓ | ✓ |
| binary | 256 | 2e-4 | ✓ | — | — | — | — |

**Missing for THU-EP FullFT:**
- [ ] 9-class B=256 lr=4e-4 Gen (s123/s456/s789/cross-seed)
- [ ] binary B=256 lr=2e-4 Gen (s123/s456/s789/cross-seed)

### 2d. THU-EP — LoRA (`--nomixup`, no `--fullft`)

| Task | B | lr | CV | Gen s123 | Gen s456 | Gen s789 | Gen cross-seed |
|------|---|----|----|----------|----------|----------|---------------|
| 9-class | 64 | 1e-4 | ✓ | ✓ | ✓ | ✓ | ✓ |
| binary | 64 | 1e-4 | ✓ | — | — | — | — |

> Also: `summary_thu-ep_9-class_w10s10_pool_last_r16_nomixup` and `summary_thu-ep_binary_w10s10_pool_last_r16_nomixup` — pool=last ablation runs.

**Missing for THU-EP LoRA:**
- [ ] binary Gen (s123/s456/s789/cross-seed)
- [ ] 9-class B=256 lr=4e-4 CV + Gen (if you want a fair comparison recipe)
- [ ] binary B=256 lr=1e-4 or 2e-4 CV + Gen

---

## 3. JADE / SupCon (`jade_checkpoints`)

All JADE runs use `--fullft` and `pool=no` unless noted. LoRA variants are absent (see missing section).

### 3a. FACED 9-class — FullFT

#### Stage 1: coarse α×τ sweep (default B=64, lr=1e-4)

| α \ τ | 0.03 | 0.05 | 0.1 | 0.2 | 0.5 |
|-------|------|------|-----|-----|-----|
| 0.1 | — | — | ✓ | — | — |
| 0.2 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 0.3 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 0.5 | — | — | ✓ | — | — |
| 0.7 | — | — | ✓ | — | — |
| 0.8 | — | — | ✓ | — | — |
| 0.9 | — | — | ✓ | — | — |

#### Stage 2: B=256 lr=4e-4 recipe (selected α×τ combos)

| α | τ | B=256 lr=4e-4 CV |
|---|---|------------------|
| 0.2 | 0.05 | ✓ |
| 0.2 | 0.1 | ✓ |
| 0.2 | 0.2 | ✓ |
| 0.2 | 0.5 | ✓ |
| 0.3 | 0.05 | ✓ |
| 0.3 | 0.1 | ✓ |
| 0.3 | 0.2 | ✓ (+ lr=1e-4, lr=2e-4, lr=8e-4 variants) |
| 0.3 | 0.5 | ✓ |
| 0.5 | 0.05 | ✓ |
| 0.5 | 0.1 | ✓ |
| 0.5 | 0.2 | ✓ |
| 0.5 | 0.5 | ✓ |

> Also run: α=0.3 τ=0.1 at B=256 lr=8e-4 (rejected for instability).

#### Generalisation (winner: α=0.3, τ=0.2, B=256, lr=4e-4)

| Gen s123 | Gen s456 | Gen s789 | Gen cross-seed |
|----------|----------|----------|---------------|
| ✓ | ✓ | ✓ | — |

Also: LoRA variant (default B=64 lr=1e-4): `summary_faced_9-class_w10s10_pool_no_r16_a0.5_t0.1_context.json` — a single old LoRA CV run + gen 3-seed + cross-seed aggregate.

**Missing for FACED 9-class JADE:**
- [ ] Gen cross-seed aggregate for winner (α=0.3 τ=0.2 B=256 lr=4e-4)
- [ ] JADE-LoRA at winner HP (α=0.3 τ=0.2 B=256 lr=4e-4) — CV + Gen
- [ ] JADE-LoRA at any B=256 recipe (currently only old default-recipe LoRA exists for α=0.5 τ=0.1)

### 3b. FACED binary — FullFT

#### Stage 1: coarse α×τ sweep (default B=64, lr=1e-4)

| α \ τ | 0.03 | 0.05 | 0.1 | 0.2 | 0.5 |
|-------|------|------|-----|-----|-----|
| 0.1 | — | — | ✓ | — | — |
| 0.2 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 0.3 | ✓ | ✓ | ✓ | ✓ | — |
| 0.5 | — | — | ✓ | — | — |
| 0.7 | — | — | ✓ | — | — |
| 0.8 | — | — | ✓ | — | — |

#### Stage 2: B=256 lr=1e-4 recipe (selected α×τ combos)

| α | τ | B=256 lr=1e-4 CV |
|---|---|------------------|
| 0.2 | 0.03 | ✓ |
| 0.2 | 0.05 | ✓ (+ lr=2e-4, 4e-4, 8e-4, 5e-5 variants) |
| 0.2 | 0.1 | ✓ (+ lr=4e-4, 8e-4 variants) |
| 0.2 | 0.2 | ✓ |
| 0.2 | 0.5 | ✓ |
| 0.3 | 0.03 | ✓ (+ lr=5e-5 variant) |
| 0.3 | 0.05 | ✓ |
| 0.3 | 0.1 | ✓ |
| 0.3 | 0.2 | ✓ |
| 0.3 | 0.5 | ✓ (+ b128 lr=1e-4 variant) |
| 0.5 | 0.03 | ✓ |
| 0.5 | 0.05 | ✓ |
| 0.5 | 0.1 | ✓ |
| 0.5 | 0.2 | ✓ |
| 0.5 | 0.5 | ✓ |
| 0.9 | — | b128 lr=1e-4 only (outlier) |

Also: `b128_lr5e-05` variant for α=0.2 τ=0.05.

#### Generalisation (winner: α=0.3, τ=0.03, B=256, lr=1e-4)

| Gen s123 | Gen s456 | Gen s789 | Gen cross-seed |
|----------|----------|----------|---------------|
| — | — | — | — |

Also older gen runs exist under different HPs:
- α=0.2 τ=0.05 b128 lr=1e-4: s123 ✓ s456 ✓ s789 ✓
- α=0.2 τ=0.1 (default recipe): s123 ✓ s456 ✓ s789 ✓ cross-seed ✓
- α=0.5 τ=0.1 (default recipe): s123 ✓ s456 ✓ s789 ✓ cross-seed ✓
- α=0.8 τ=0.1 (default recipe): s123 ✓ s456 ✓ s789 ✓ cross-seed ✓

Also: LoRA variants (old default recipe): `a0.2_t0.1_context` and `a0.5_t0.1_context` and `a0.8_t0.1_context` — CV + gen 3-seed + cross-seed.

**Missing for FACED binary JADE:**
- [ ] Gen (s123/s456/s789/cross-seed) for winner α=0.3 τ=0.03 B=256 lr=1e-4
- [ ] JADE-LoRA at winner HP (α=0.3 τ=0.03 B=256 lr=1e-4) — CV + Gen

### 3c. THU-EP — FullFT (cross-dataset transfer, secondary)

| Task | α | τ | B | lr | CV |
|------|---|---|---|----|----|
| 9-class | 0.3 | 0.2 | 256 | 4e-4 | ✓ |
| binary | 0.3 | 0.03 | 256 | 1e-4 | ✓ |

**Missing for THU-EP JADE:**
- [ ] Everything else — treated as secondary, no further runs planned per thesis scope

---

## 4. Summary of Missing Runs (Priority Order)

### High priority (thesis-critical)

| # | Approach | Dataset | Task | Config | What's missing |
|---|----------|---------|------|--------|----------------|
| 1 | JADE-FullFT | FACED | 9-class | α=0.3 τ=0.2 B=256 lr=4e-4 | Gen cross-seed aggregate |
| 2 | JADE-FullFT | FACED | binary | α=0.3 τ=0.03 B=256 lr=1e-4 | Gen s123/s456/s789 + cross-seed |
| 3 | FT-LoRA | FACED | 9-class | B=256 lr=4e-4 | CV + Gen |
| 4 | FT-LoRA | FACED | binary | B=256 lr=1e-4 | CV + Gen |
| 5 | JADE-LoRA | FACED | 9-class | α=0.3 τ=0.2 B=256 lr=4e-4 | CV + Gen |
| 6 | JADE-LoRA | FACED | binary | α=0.3 τ=0.03 B=256 lr=1e-4 | CV + Gen |

### Medium priority (completeness / fair comparison)

| # | Approach | Dataset | Task | Config | What's missing |
|---|----------|---------|------|--------|----------------|
| 7 | FT-FullFT | FACED | 9-class | B=256 lr=4e-4 | Gen cross-seed aggregate |
| 8 | FT-FullFT | FACED | binary | B=256 lr=1e-4 | Gen s123/s456/s789 + cross-seed |
| 9 | FT-FullFT | FACED | binary | B=256 lr=2e-4 | Gen s123/s456/s789 + cross-seed |
| 10 | FT-FullFT | THU-EP | 9-class | B=256 lr=4e-4 | Gen s123/s456/s789 + cross-seed |
| 11 | FT-FullFT | THU-EP | binary | B=256 lr=2e-4 | Gen s123/s456/s789 + cross-seed |
| 12 | FT-LoRA | THU-EP | binary | default | Gen s123/s456/s789 + cross-seed |
