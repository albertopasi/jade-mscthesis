# Architecture & Code Design

How the training and inference pipelines are wired together. This is the doc you read **before** modifying the code — it explains the conventions that the rest of the codebase silently relies on, and why the abstractions are shaped the way they are.

If you only want to *run* things, read the repo [`README.md`](../README.md) and [`reproducing_results.md`](reproducing_results.md) instead. This doc is about the *internals*.

---

## High-level picture

```
       raw EEG (.pkl / .mat)
                │
                ▼
    src/preprocessing/  (resample 250→200 Hz, .npy on disk)
                │
                ▼
      ┌──────────────────────────┐
      │  src/datasets/           │
      │  EEGWindowDataset        │  ← serves (window, label) batches
      └──────────────────────────┘
                │
                ▼
      ┌──────────────────────────┐
      │  src/approaches/         │
      │   ├── linear_probing/    │
      │   ├── fine_tuning/       │  three pipelines, one shared training
      │   └── jade/              │  loop scaffolding (Stage 1 / Stage 2)
      └──────────────────────────┘
                │
                ▼
      ┌──────────────────────────┐
      │  src/inference/          │
      │   ├── predictions.py     │  run_fold_inference  ──► FoldPredictions
      │   └── aggregate.py       │  write_run_summary   ──► main-results/<approach>_<task>/<stem>.json
      └──────────────────────────┘
                │
                ▼
   src/inference/statistical_tests*.py  →  docs/statistical_tests*.md
   src/visualization/make_*.py         →  src/visualization/jade_<task>/figures/
```

There are three training scripts (one per approach), but they all funnel into a **single inference contract** (`FoldPredictions`) and a **single on-disk schema** (`write_run_summary`). Downstream analysis (statistical tests, plots, paper tables) never cares which training pipeline produced a result.

---

## Three training pipelines, one shape

[`src/approaches/`](../src/approaches/) has three subpackages — `linear_probing/`, `fine_tuning/`, `jade/` — plus a `shared/` package for the parts that don't change between approaches.

The three pipelines deliberately share **identical conventions**:

| Convention | Where it lives |
|---|---|
| Config dataclass with `run_name()` / `run_name_stem()` helpers | `<approach>/config.py` |
| `run_fold_*()` per-fold runner returning a result dict | `<approach>/train_*.py` |
| Two-stage training (LP warmup → main stage) for FT and JADE | `<approach>/training.py` |
| Best-state capture inside the inner training loop | `train_stage*()` in `training.py` |
| End-of-fold inference + summary writing | wired in the per-fold runner |
| W&B project per approach (`eeg-lp-v2`, `eeg-ft-v2`, `eeg-jade-v2`) | `<approach>/config.py` |
| Output dirs (`outputs/<approach>_checkpoints/`) | `<approach>/config.py` |

If you're adding a fourth approach, **copy `src/approaches/jade/` as a starting point**. It has all the moving parts the other two have, plus the projection-head and SupCon logic that would otherwise be a recurring source of subtle bugs to re-derive.

### Two-stage training (FT, JADE)

Both SFT and JADE use the same two-stage schedule:

- **Stage 1 (LP warmup):** encoder frozen, only `cls_query_token` + linear head trainable, cross-entropy loss, LR 5e-3, ~20 epochs max. JADE also has the projection head present architecturally but frozen and unused.
- **Stage 2 (main):** encoder unfrozen (LoRA adapters or full encoder, controlled by `--fullft`). LR 1e-4 by default, up to 200 epochs, early stop patience 20. For JADE the projection head activates and the SupCon term joins the loss: `L = λ · L_CE + (1 − λ) · L_SupCon`.

Both stages reuse `train_stage()` (or `train_stage_jade()` for JADE), which is the same inner loop with different config: AMP autocast, StableAdamW, exponential warmup → ReduceLROnPlateau, mixup-with-CE for the non-SupCon path, gradient clip 2.0, AMP float16 throughout.

**Critical norm:** when JADE departs from the REVE recipe (e.g. B=256 with scaled LR), the SFT baseline must be re-run at the *same* recipe for a fair comparison. JADE@(new recipe) vs SFT@(old recipe) is not a valid comparison. The HP tuning protocol in [`jade_hp_sweep.md`](jade_hp_sweep.md) is built to prevent this.

### LoRA vs full FT

`--fullft` is the kill switch:

- **Without `--fullft`**: the Stage-2 encoder gets LoRA adapters injected via `peft` (`apply_lora`, [`src/approaches/fine_tuning/lora.py`](../src/approaches/fine_tuning/lora.py)). Target modules are `transformer.layers.{i}.0.to_qkv` and `to_out` across all 22 layers. Default rank=16, alpha=16. Checkpoint saves the LoRA adapter dir + head weights separately.
- **With `--fullft`**: no peft wrapper; the encoder is unfrozen as a single nn.Module. Checkpoint saves `full_model.pt` — the entire state dict.

The thesis winning configs are all `--fullft`. LoRA is a memory-efficient alternative and is the right default when GPU RAM is tight.

### Best-state restore — load order matters

Each stage's `train_stage_*()` captures the best epoch's state dict in CPU memory as `best_state` and returns it in the result dict. The per-fold runner then **restores it before anything that depends on the best model**:

```python
# train_jade.py (and analogous in train_ft.py, train_lp.py)
if ft_result.get("best_state"):
    model.load_state_dict(ft_result["best_state"])
    model.to(DEVICE)

# Only AFTER restore: run the in-training inference + (optionally) save checkpoint
fold_predictions = run_fold_inference(model, val_ds, ...)
if cfg.save_checkpoints:
    torch.save(...)
```

If you ever skip the restore, end-of-fold inference will silently use whatever weights the **last** training step produced, not the best epoch. That's been a bug source in older versions of the code — keep the restore line.

---

## REVE integration

REVE is loaded once per script invocation (not per fold) via `load_reve_and_positions()` in [`src/approaches/shared/reve.py`](../src/approaches/shared/reve.py). The returned `reve_model` and `pos_tensor` are then **deep-copied per fold** inside each `run_fold_*()`:

```python
reve_for_fold = copy.deepcopy(reve_model)
pos_for_fold = pos_tensor.clone()
```

Why: `apply_lora()` mutates the encoder in-place (via `peft.get_peft_model`), and full FT updates encoder weights directly. Without the per-fold deep-copy, fold 2 starts from fold 1's adapted weights — silent leakage across folds.

### REVE quirks worth remembering

- **`cls_query_token` is a trainable parameter even under "frozen encoder" mode.** It lives outside the encoder and is what LP actually trains alongside the linear head. Both LP checkpoint files contain `cls_query_token.*` keys; missing this is how you get LP runs that converge to chance.
- **Position tensor lives outside the encoder.** REVE forward takes `(eeg, pos)`. The classifier wrappers (`ReveClassifierLP`, `ReveClassifierFT`, `ReveClassifierJADE`) register `pos_tensor` as a buffer so calling `model(eeg)` is sufficient downstream.
- **`n_patches = floor((window_size − patch_size) / (patch_size − overlap)) + 1`**. For window_size=2000, patch_size=200, overlap=20, this gives `n_patches = 11`. Used by the `pooling=no` head dimension calc.
- **Scale factor 1000 is applied at dataset `__getitem__` time, not at preprocessing.** Preprocessed `.npy` files store raw µV. The `scale_factor` divisor lives in [`EEGWindowDataset`](../src/datasets/base.py).
- **Pooling modes** — `no` (default, used in the winning configs) concats `[query, patches]` and flattens; `last` returns the query attention output squeezed; `last_avg` mean-pools all patches. `no` has the largest classifier (180,736 input dim on FACED) but the best performance.

---

## Config conventions: `run_name` and `run_name_stem`

Every config dataclass exposes two helpers:

- **`run_name_stem(gen_seed=None)`** — the stable identifier shared across folds. Encodes every hyperparameter that distinguishes a sweep cell.
- **`run_name(fold_idx, gen_seed=None)`** — `<run_name_stem>_fold_<K>`. Per-fold checkpoint dir.

```python
# Example, JADE 9-class winning config:
cfg.run_name_stem()
# → 'jade_faced_9-class_w10s10_pool_no_r16_a0.3_t0.2_context_b256_lr0.0004_fullft'

cfg.run_name(fold_idx=1)
# → 'jade_faced_9-class_..._fullft_fold_1'

cfg.run_name_stem(gen_seed=123)
# → 'jade_faced_9-class_..._fullft_gen_s123'

cfg.run_name(fold_idx=1, gen_seed=123)
# → 'jade_faced_9-class_..._fullft_gen_s123_fold_1'
```

This single source of truth means:

- Checkpoint dir: `outputs/<approach>_checkpoints/<run_name>/`
- W&B run name: `<run_name>`
- Main-results JSON: `main-results/<approach>_<task>/<run_name_stem>.json`
- Generalization JSON: `main-results/<approach>_<task>_generalization/<run_name_stem>.json` (the seed is already in the stem)
- Cross-seed aggregate: `main-results/<approach>_<task>_generalization/<common_stem>_gen_avg.json`

If you add a new hyperparameter that distinguishes sweep cells (e.g. a new loss term), it **must** be encoded in `run_name_stem`. Otherwise two different runs collide on disk and you silently overwrite results.

---

## The inference contract

Inference happens in two contexts that look superficially different but go through the same code path:

1. **Post-hoc inference**: load each fold's checkpoint from disk, build the val dataset, forward, aggregate. Driver: [`src/inference/inference_subject_wise.py`](../src/inference/inference_subject_wise.py).
2. **In-training inference**: at the end of each fold (just after the best-state restore), forward the val set in memory, aggregate. Used for the stimulus-generalization runs because we deliberately don't save checkpoints (`--no-save-checkpoints`) — 4 jobs × 3 seeds × 10 folds = 120 checkpoints would be wasteful when we only need predictions.

Both contexts use the **same two functions** to produce **the same JSON+NPZ output**:

### `FoldPredictions` + `run_fold_inference()` — single inference primitive

Defined in [`src/inference/predictions.py`](../src/inference/predictions.py).

```python
@dataclass
class FoldPredictions:
    fold: int | None
    gen_seed: int | None
    val_subject_ids: list[int]
    val_stimuli: list[int] | None
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray
    subj_ids: np.ndarray
    stim_ids: np.ndarray
    window_starts: np.ndarray
    per_subject_acc: dict[int, float]
    per_subject_support: dict[int, int]
    window_acc: float
```

This dataclass is **the** unit of inference output. Both call sites produce one. Downstream code never reaches into a dataset or a model — it only reads `FoldPredictions`.

`run_fold_inference()` itself is short (~50 LOC): build a DataLoader with `shuffle=False, num_workers=0`, forward each batch under autocast + no_grad, concatenate, then recover per-window subject/stimulus IDs from `val_ds.index[i]`.

> **Critical**: `num_workers=0` is intentional. With `num_workers > 0`, PyTorch workers do prefetching and the batch order is no longer guaranteed to match `val_ds.index` index order. That would scramble per-subject accuracies silently. Don't change it.

### `write_run_summary()` — single output-schema owner

Defined in [`src/inference/aggregate.py`](../src/inference/aggregate.py).

Takes a list of `FoldPredictions` covering all folds of a (method, task) run, pools per-window arrays, computes the confusion matrix + per-class P/R/F1 + AUROC, and writes JSON + NPZ in the canonical schema. The function is the **only** place that knows the on-disk schema — everything downstream (`statistical_tests.py`, the figure scripts, the seed-averaging script) reads what this function writes.

The schema:

```
main-results/<approach>_<task>[_generalization]/<run_stem>.json
main-results/<approach>_<task>[_generalization]/<run_stem>.npz
```

JSON top-level keys:

```
approach, task, dataset, run_stem, completed_at,
n_folds_run, n_subjects, gen_seed,
subject_wise: {mean_acc, std_acc, min_acc, max_acc},
window_wise_acc,
classification_report: {labels, confusion_matrix, per_class, macro, auroc_per_class},
per_subject_acc: {sid → acc},   ← the vector used by statistical_tests.py
per_subject_support: {sid → n_windows},
folds: [{fold, gen_seed, val_subjects, val_stimuli, window_acc, macro_subject_acc}, ...]
extra: {...}   ← optional, used by training-time writes to include HPs
```

NPZ arrays: `y_true, y_pred, y_prob, subj_ids, stim_ids, labels`.

### Seed averaging (generalization runs)

For generalization runs we train with multiple stimulus-split seeds (default `123 456 789`). Each seed produces one `<stem>_gen_s<seed>.json` + `.npz`. To get the canonical "averaged" result, run [`src/inference/average_gen_seeds.py`](../src/inference/average_gen_seeds.py):

```bash
uv run python -m src.inference.average_gen_seeds --approach jade --task 9-class
```

This globs the per-seed JSONs, averages each subject's accuracies across seeds (keeping subject as the unit of analysis), pools predictions across seeds for the macro report, and writes `<common_stem>_gen_avg.json`. The averaged JSON has the same schema as a normal one, with two extra fields (`gen_seeds`, `n_seeds_per_subject`).

---

## Statistical tests

Two driver scripts, both consume `main-results/<approach>_<task>/<stem>.json`:

- **[`statistical_tests.py`](../src/inference/statistical_tests.py)** — the main-results protocol. Per-condition BCa CI on the mean, paired Wilcoxon (primary) + paired t (companion) + sign test (convergent) on the JADE-vs-baseline pairs, Holm correction across the 4-pair family. Output: [`docs/statistical_tests.md`](statistical_tests.md).
- **[`statistical_tests_generalization.py`](../src/inference/statistical_tests_generalization.py)** — the generalization protocol. Friedman omnibus across LP/SFT/JADE, post-hoc Wilcoxon (only if omnibus rejects), Brown-Forsythe-via-Friedman dispersion omnibus, BCa CIs on variance ratios. Output: [`docs/statistical_tests_generalization.md`](statistical_tests_generalization.md).

Both files import shared helpers from `statistical_tests.py` so the two documents stay consistent (same BCa CI definition, same Holm correction, same formatting). If you change `bca_bootstrap_ci()` or `holm_bonferroni()`, both reports update on re-run. A short explanation of the BCa vs percentile distinction lives inline in [`statistical_tests.md`](statistical_tests.md) §0.4.

---

## Visualization

[`src/visualization/`](../src/visualization/) is a thin layer that reads the canonical JSONs/NPZs and writes PDFs to `src/visualization/jade_<task>/figures/`. Output is **PDF only** (vector, for thesis insertion).

Convention: every figure script is a `make_<thing>.py`, takes `--task {9-class, binary}`, loads via `_common.load_run()`, and saves via `_common.save_fig()`. [`make_all.py`](../src/visualization/make_all.py) runs them all per task.

The single exception is [`plot_violins.py`](../src/visualization/plot_violins.py), which doesn't take `--task` because it deliberately produces figures for both tasks in one invocation.

To regenerate every figure:

```bash
uv run python -m src.visualization.make_all
uv run python -m src.visualization.plot_violins
```

---

## SLURM

All long-running jobs go through SLURM. The generic launcher is [`slurm/run_experiment.sh`](../slurm/run_experiment.sh): one job = one `uv run python -m <module> <args>`. Per-job time, partition, and account are baked into the script header.

Higher-level orchestration (sweeps that submit multiple jobs) lives in `slurm/run_*.sh`. The two important ones:

- [`slurm/run_jade_bulletproof.sh`](../slurm/run_jade_bulletproof.sh) — main-results sweep that produced the winning configs.
- [`slurm/run_faced_generalization.sh`](../slurm/run_faced_generalization.sh) — stimulus-generalization sweep.

The `run_experiment.sh` launcher already exports `WANDB_MODE=offline` (DelftBlue GPU nodes have no internet). After the job finishes, `wandb sync` from a login node.

---

## What I deliberately don't abstract

- **Training-loop boilerplate.** Each pipeline has its own `training.py` with its own `train_stage()` function. They share a lot but the differences (SupCon insertion, mixup gating, projection head freeze/unfreeze) are subtle enough that "DRY" abstraction would hurt more than it helps. The duplication is intentional.
- **Loss functions.** SupCon lives only in `src/approaches/jade/loss.py`. No `src/losses/` directory; SupCon isn't reused outside JADE.
- **Per-approach W&B logging.** Each pipeline calls `wandb.log()` directly — no shared logger. Different approaches want to log different things (component losses for JADE, LoRA stats for FT, etc.).

If you find yourself wanting to abstract any of these, **don't** unless you have at least three concrete call sites that genuinely share the same shape. The current duplication is cheap; premature abstraction is what makes this kind of codebase hard to read in 6 months.

---

## Useful entry points by task

| Task | Start reading at |
|---|---|
| Add a new approach | [`src/approaches/jade/train_jade.py`](../src/approaches/jade/train_jade.py) (copy as template) |
| Add a new figure | [`src/visualization/make_subject_histogram.py`](../src/visualization/make_subject_histogram.py) (minimal template) |
| Add a new statistical test | [`src/inference/statistical_tests.py`](../src/inference/statistical_tests.py) (`run_paired` and `run_friedman` are the patterns) |
| Change the JSON schema | [`src/inference/aggregate.py`](../src/inference/aggregate.py) — and update [`statistical_tests.py`](../src/inference/statistical_tests.py) to read the new fields |
| Add a new dataset | [`src/datasets/faced_dataset.py`](../src/datasets/faced_dataset.py) (subclass `EEGWindowDataset`) and a corresponding `src/preprocessing/<dataset>/run_preprocessing.py` |
| Wire something new into config | [`src/approaches/<approach>/config.py`](../src/approaches/jade/config.py) — and remember to encode it in `run_name_stem` if it distinguishes sweep cells |
