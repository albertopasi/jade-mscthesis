# Test Coverage

This doc inventories the test suite: what is covered, by which file, and what
deliberately is not. CI runs every file listed here except `test_reve.py`
(skipped in CI because it needs the REVE weights, which aren't shipped in the
repo).

Run everything locally:

```bash
uv run python -m pytest tests/ -v               # full suite (needs REVE for test_reve.py)
uv run python -m pytest tests/ --ignore=tests/test_reve.py -v   # CI configuration
```

**Headline numbers:** 252 tests passing in CI configuration, full suite
in ~45 s, all tests CPU-only (no GPU required).

---

## Coverage map

| Test file | Tests | Covers | Module(s) under test |
|---|---:|---|---|
| [`test_statistical_helpers.py`](../tests/test_statistical_helpers.py) | 28 | Pure-math helpers behind `docs/statistical_tests*.md` | [`src/inference/statistical_tests.py`](../src/inference/statistical_tests.py) |
| [`test_run_paired.py`](../tests/test_run_paired.py) | 33 | Paired-test driver (Wilcoxon, t, sign, Cohen's d, rank-biserial, diagnostics) | [`src/inference/statistical_tests.py`](../src/inference/statistical_tests.py) |
| [`test_config_naming.py`](../tests/test_config_naming.py) | 37 | `cfg.run_name_stem()` / `cfg.run_name()` collision prevention for all three pipelines | [`src/approaches/{lp,ft,jade}/config.py`](../src/approaches/jade/config.py) |
| [`test_supcon_loss.py`](../tests/test_supcon_loss.py) | 22 | JADE's contrastive loss (Khosla `L_sup_out`) | [`src/approaches/jade/loss.py`](../src/approaches/jade/loss.py) |
| [`test_predictions_aggregate.py`](../tests/test_predictions_aggregate.py) | 20 | Inference plumbing + JSON+NPZ schema | [`src/inference/predictions.py`](../src/inference/predictions.py), [`src/inference/aggregate.py`](../src/inference/aggregate.py) |
| [`test_datasets.py`](../tests/test_datasets.py) | 33 | EEG window datasets, label maps, stimulus filtering | [`src/datasets/`](../src/datasets/) |
| [`test_folds.py`](../tests/test_folds.py) | 28 | k-fold + stimulus generalization + official split | [`src/datasets/folds.py`](../src/datasets/folds.py) |
| [`test_preprocessing_faced.py`](../tests/test_preprocessing_faced.py) | 17 | FACED `.pkl` → `.npy` (resampling, shapes, channel layout) | [`src/preprocessing/faced/`](../src/preprocessing/faced/) |
| [`test_preprocessing_thu_ep.py`](../tests/test_preprocessing_thu_ep.py) | 34 | THU-EP `.mat` → `.npy` (band extraction, channel removal, z-norm, clipping) | [`src/preprocessing/thu_ep/`](../src/preprocessing/thu_ep/) |
| [`test_reve.py`](../tests/test_reve.py) | 28 | REVE loading + shared utilities | [`src/approaches/shared/`](../src/approaches/shared/) (skipped in CI — needs weights) |

**Total:** 280 tests across 10 files. 252 run in CI; `test_reve.py` (28 tests) runs locally only.

---

## What each file pins

### `test_statistical_helpers.py` (28 tests)

Covers every pure-math function whose output ends up in
`docs/statistical_tests*.md`:

- **`holm_bonferroni`** — output for the classical hand-computable example,
  cap at 1.0, monotonicity across 20 random inputs, behaviour for unsorted
  input (output must align with input positions, not sorted positions),
  edge cases (`m=1`, extremely small p-values).
- **`bca_bootstrap_ci`** — reproducibility under fixed seed, divergence
  under different seeds, narrowing as N grows, **reference check against a
  direct `scipy.stats.bootstrap(..., method='BCa')` call** with the same
  seed.
- **`bca_variance_ratio_ci`** — equal-variance inputs give a CI bracketing
  1.0, larger-variance-in-a gives ratio > 1, mismatched lengths raise.
- **`run_friedman`** — reference check against `scipy.stats.friedmanchisquare`,
  strong-signal data rejects H₀, mean-rank ordering matches the data,
  Kendall's W formula `W = χ² / (n·(k−1))`, requires `k ≥ 3`.
- **`run_brown_forsythe_friedman`** — equal-dispersion → does not reject,
  unequal-dispersion → rejects, does not mutate input.

### `test_run_paired.py` (33 tests)

Covers every field of `PairedResult` and the `fmt_p` formatter:

- **Shape + counts**: `mean_a`/`mean_b`/`mean_diff`/`median_diff` exact
  match, `std_diff` uses `ddof=1`, `wins + losses + ties = n`, ties detected.
- **Parametric block**: t-statistic + p-value match `scipy.stats.ttest_rel`,
  t-based CI brackets `mean_diff` when significant, formula
  `CI = mean_diff ± t_crit · std_diff / sqrt(n)`.
- **Wilcoxon block**: matches `scipy.stats.wilcoxon(zero_method="wilcox")`,
  rejects strong synthetic signal.
- **Sign test**: nan when all tied, tiny p under extreme dominance,
  matches `scipy.stats.binomtest`.
- **Effect sizes**: Cohen's d formula + sign, rank-biserial bounded in
  [−1, +1], reaches 1.0 under extreme dominance, nan when all tied.
- **Distributional diagnostics**: skew, excess kurtosis, Shapiro p — all
  match scipy directly.
- **Reproducibility**: bootstrap CI is seed-deterministic.
- **`fmt_p`**: nan → `—`, very small → scientific notation, `< 0.001` use
  the marker `<0.001`.

### `test_config_naming.py` (37 tests)

The collision-prevention test. If a future commit adds an HP but forgets to
encode it in `run_name_stem`, two sweep cells silently overwrite each
other on disk. These tests catch that:

- **Per-approach** (LP, SFT, JADE): prefix correctness, `_fold_K` suffix
  from `run_name`, generalization seed embedding (`_gen_s{seed}`), key tag
  presence (`_nomixup`, `_fullft`).
- **Thesis-final stems pinned exactly** for JADE 9-class, JADE binary, FT
  9-class, LP 9-class. If these strings ever change, the thesis JSONs no
  longer round-trip cleanly.
- **`test_each_hp_distinguishes_stem`** (parametrized): flips one HP at a
  time and asserts the stem changes. Currently covers `batch_size`,
  `ft_lr`, `lora_rank`, `pooling`, `use_mixup`, `full_ft`, `supcon_alpha`,
  `supcon_temperature`, `supcon_repr`, `task_mode`, `normalize_features`.
- **Cross-approach invariants**: LP/SFT/JADE prefixes never collide; no
  double/trailing underscores; folds are 1-indexed.

### `test_supcon_loss.py` (22 tests)

The actual training signal of JADE. A bug here changes what gets learned.

- **Shape + scalars**: returns a scalar tensor, always non-negative
  (negative log-prob average), always finite.
- **Singleton handling**: all-singletons → exact 0.0 loss that still
  permits `.backward()` (training-loop safety contract); singletons in a
  mixed batch are excluded from the per-anchor mean.
- **Invariances**: relabelling integer class IDs (0,1,2 → 7,3,5) gives
  identical loss; shuffling `(z, y)` rows together is invariant.
- **Numerical reference**: hand-computed value for a 4-sample, 2-class
  case where the math collapses cleanly (`log(1 + 2·exp(-2))`).
- **`log(|P(i)|)` floor**: with perfectly clustered features, the loss
  does *not* go to 0 — it bottoms out at `log(|P|)` because positives
  compete with each other in the softmax denominator. Pinned so a future
  "optimization" doesn't silently remove this property.
- **Temperature**: changes loss, stored on the module, default is 0.1.
- **Gradient flow**: `z.grad` is non-zero after `.backward()` on a normal
  batch; `z.grad` is `None` (not zero) when all classes are singletons
  because the loss returns a fresh zero tensor unattached to `z`'s graph.
- **Numerical stability**: finite across τ ∈ {0.01, 0.05, 0.1, 0.5, 1.0},
  finite at batch size 256, no overflow at small τ.

### `test_predictions_aggregate.py` (20 tests)

Covers the inference primitive and the on-disk schema. Uses a stub
dataset + stub model (no REVE, no real EEG).

- **`run_fold_inference`**:
  - Raises `TypeError` when the dataset has no `.index` attribute.
  - Per-window `subj_ids` / `stim_ids` / `window_starts` recovered from
    `val_ds.index` match the dataset's index column.
  - **Order invariant**: with `shuffle=False, num_workers=0`, the
    batch-concatenated output order matches `val_ds.index` 1:1 (the
    contract that prevents per-subject accuracies from being scrambled).
  - Per-subject accuracy = mean of `(y_pred == y_true)` over that
    subject's windows; subjects not in the dataset are silently skipped.
  - Per-subject support reports correct window counts.
- **`write_run_summary`**:
  - Empty `fold_preds` raises `ValueError`.
  - Subject appearing in two folds raises `RuntimeError` (the
    cross-subject CV invariant).
  - **JSON schema pinned**: top-level keys (`approach`, `task`, `dataset`,
    `run_stem`, `n_folds_run`, `n_subjects`, `gen_seed`, `subject_wise`,
    `window_wise_acc`, `classification_report`, `per_subject_acc`,
    `per_subject_support`, `folds`).
  - `per_subject_acc` keys are JSON-safe strings castable back to int.
  - `subject_wise` block has correct mean/std/min/max with `ddof=1`.
  - `classification_report` contains labels, square confusion matrix,
    per-class P/R/F1, macro averages.
  - `extra_metadata` persists to the JSON.
  - Generalization mode writes to `<approach>_<task>_generalization/` and
    records `gen_seed` in the JSON body.
  - NPZ has the expected array names (`y_true`, `y_pred`, `y_prob`,
    `subj_ids`, `stim_ids`, `labels`) with aligned lengths.
  - Output filename is exactly `<run_stem>.json` (no double-encoding of
    `gen_seed`).
- **End-to-end round-trip**: build two folds → run inference → write
  summary → read back → confirm 4 subjects, perfect mean accuracy.

### `test_datasets.py` (33 tests)

EEG window datasets — already comprehensive (predates the refactor).
Covers FACED + THU-EP window construction, label-map round-trip,
stimulus filtering, the `EEGWindowDataset.index` contract, excluded
subjects/stimuli, `build_raw_dataset` factory.

### `test_folds.py` (28 tests)

Fold-splitting primitives — already comprehensive. Covers k-fold
disjointness, "every subject in exactly one val fold" invariant, the
stimulus-generalization split (different seeds, train/val disjoint,
all 28 stimuli covered, balanced per emotion), official REVE split,
the `N_FOLDS` constant.

### `test_preprocessing_faced.py` (17 tests)

`.pkl` → `.npy` for FACED. Covers resampling (scipy direct, mean
preservation, ratio, identity at same rate, already-resampled
passthrough), output shape and dtype, no-NaN/no-Inf guarantee, validate
mode, missing-file handling.

### `test_preprocessing_thu_ep.py` (34 tests)

`.mat` → `.npy` for THU-EP. Covers frequency-band extraction (broad-band
index, all valid indices, correct band extracted), channel removal (`A1`,
`A2` indices, final channel count), downsampling (sampling properties,
matches scipy), z-normalization (preserves mean, normalized stats,
zero-std channel safety), artifact clipping (threshold behaviour,
no-clipping within range, statistics returned), export (file existence,
dtype, shape), config loading (from YAML, missing config raises,
steps-enabled toggle), end-to-end pipeline shape integrity.

### `test_reve.py` (28 tests, **CI-excluded**)

Needs the REVE weights, which aren't in git. Run locally only. Covers
channel-name resolution per dataset, `load_reve_and_positions`, RMSNorm
output shape and statistics, `compute_n_patches` formula, shared config
constants.

---

## What deliberately isn't tested

These are documented gaps — not bugs in the suite, but conscious choices.

- **End-to-end training smoke tests.** Even one fold with 2+3 epochs takes
  minutes and requires REVE weights and a GPU. Smoke tests live in
  `slurm/run_gen_smoke.sh` for manual verification before submitting
  expensive sweeps. CI doesn't run them.
- **Visualization scripts** (`src/visualization/make_*.py`). Figures are
  inspected visually. A pixel-perfect test on a PDF would be brittle and
  add nothing.
- **`statistical_tests.py` main markdown renderer**. The `main()` function
  reads real `main-results/*.json` files and writes
  `docs/statistical_tests.md`. The component pieces (`holm_bonferroni`,
  `bca_bootstrap_ci`, `run_paired`, `run_friedman`, etc.) are tested; the
  markdown-string-assembly layer is treated as plumbing.
- **`average_gen_seeds.py`**. Wraps the same `predictions` + `aggregate`
  primitives that *are* tested. Could be added if the seed-averaging
  logic gets more complex.
- **`run_fold_inference` empty-dataset behaviour**. Currently raises
  `ValueError` from `np.concatenate([])`. Not a tested edge case because
  it's not expected to happen in practice — the training loop never
  feeds an empty val set.
- **GPU code paths**. All tests run on CPU. CUDA kernel correctness is
  Torch's responsibility, not ours.

---

## CI integration

GitHub Actions runs the CI configuration (excluding `test_reve.py`) on
every push and PR. See [`.github/workflows/`](../.github/workflows/) for
the exact workflow definition. Ruff lint and format are also checked.

To replicate locally:

```bash
uv run python -m pytest tests/ --ignore=tests/test_reve.py -v --tb=short
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

All three must pass for a clean PR.
