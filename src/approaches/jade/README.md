# JADE — Joint Alignment and Discriminative Embedding

Joint Cross-Entropy + Supervised Contrastive fine-tuning of the REVE encoder
for EEG emotion classification.

Stage 1 (LP warmup): frozen encoder, train `cls_query_token` + linear head with CE only — identical to LP / FT warmup.
Stage 2 (FT): unfreeze encoder (LoRA by default, or full FT) and optimise the joint loss

```
L = α · CE(logits, y)  +  (1 − α) · SupCon(z, y)
```

where `z` are L2-normalised projections produced by a small MLP head attached
to a compact pooled representation of the encoder output.

Motivation: the FT results (`docs/ft_results_analysis.md`) show that CE-only
fine-tuning gets higher cross-subject CV accuracy but collapses on stimulus
generalisation — it overfits stimulus-specific cues. SupCon explicitly pulls
same-emotion embeddings together across stimuli and is the natural remedy.

---

## Files

| File | Description |
|---|---|
| `config.py` | `JADEConfig` dataclass — paths, hyperparameters, LoRA settings, SupCon hyperparameters, derived helpers |
| `model.py` | `ReveClassifierJADE` — FT architecture + projection head for SupCon |
| `loss.py` | `SupConLoss` — Khosla et al. 2020 supervised contrastive loss (L_sup_out) |
| `training.py` | `train_stage_jade()` — shared loop for LP warmup (CE only) and FT (joint CE + SupCon) |
| `summary.py` | Thin wrapper around `shared/summary.py` for JADE-specific filenames |
| `train_jade.py` | Main entry point — CLI, fold orchestration, checkpoint saving |

LoRA injection is reused from `src/approaches/fine_tuning/lora.py` (`apply_lora` is duck-typed on `cfg.lora_*` attributes, which `JADEConfig` provides).

---

## Evaluation modes (mutually exclusive)

| Flag | Mode | Subjects split | Output |
|---|---|---|---|
| *(default)* | 10-fold cross-subject CV | `KFold(n=10, seed=42)` | `summary_*.json` |
| `--revesplit` | Official REVE static split (FACED only) | train 0–79 / val 80–99 / test 100–122 | `summary_*_revesplit.json` |
| `--generalization` | Stimulus generalisation | same k-fold subjects, 2/3 train stimuli / 1/3 held-out, 3 seeds | `summary_*_gen_s{seed}.json` + `summary_*_gen_cross_seed.json` |

---

## Stage 2 modes

| Flag | Behaviour |
|---|---|
| *(default)* | LoRA adapters injected into `to_qkv` + `to_out` of all 22 transformer layers |
| `--fullft` | Entire encoder unfrozen, no LoRA — all encoder parameters updated |

---

## Running

```bash
# All folds, FACED, 9-class, LoRA + SupCon (default)
uv run python -m src.approaches.jade.train_jade --dataset faced --task 9-class

# Single fold
uv run python -m src.approaches.jade.train_jade --dataset faced --fold 1

# Custom SupCon weighting
uv run python -m src.approaches.jade.train_jade --dataset faced --alpha 0.7 --temperature 0.1

# Projection-head input representation
uv run python -m src.approaches.jade.train_jade --dataset faced --supcon-repr mean    # or: context (default), both

# Full fine-tuning (no LoRA) + SupCon
uv run python -m src.approaches.jade.train_jade --dataset faced --fullft

# Official REVE static split with held-out test evaluation
uv run python -m src.approaches.jade.train_jade --dataset faced --revesplit

# Stimulus generalisation (3 seeds by default)
uv run python -m src.approaches.jade.train_jade --dataset faced --generalization

# Smoke test
uv run python -m src.approaches.jade.train_jade --dataset faced --fold 1 --lp-epochs 2 --ft-epochs 3
```

---

## Training details

| Setting | LP warmup stage | FT + SupCon stage |
|---|---|---|
| Loss | CE only | α · CE + (1 − α) · SupCon |
| Optimiser | StableAdamW (β=[0.92, 0.999], wd=0.01) | same |
| LR | 5e-3 | 1e-4 |
| LR schedule | Exp warmup (3 ep) + ReduceLROnPlateau (patience=6) | Exp warmup (5 ep) + ReduceLROnPlateau (patience=6) |
| Max epochs | 20 | 200 |
| Early stopping | patience=10 on val_acc | patience=20 on val_acc |
| Grad clip | max_norm=2.0 | max_norm=2.0 |
| Dropout (head) | 0.05 | 0.1 |
| Batch size | 128 | 128 |
| Precision | AMP float16 | AMP float16 |
| Mixup | disabled | disabled (SupCon requires clean labels) |
| Projection head | frozen | trainable |
| Best model metric | val_acc | val_acc |

Batch size is 128 (vs 64 in FT) because SupCon's gradient signal scales with
the number of same-class positives per anchor — denser batches give a
lower-variance contrastive signal. The CE component is unaffected.

---

## SupCon configuration

| Flag | Default | Description |
|---|---|---|
| `--alpha` | 0.5 | CE weight: `L = α·CE + (1−α)·SupCon` |
| `--temperature` | 0.07 | SupCon temperature τ |
| `--proj-dim` | 128 | Projection head output dim |
| `--proj-hidden` | 512 | Projection head hidden dim |
| `--supcon-repr` | `context` | Projection input: `context` (query-attention, 512-d), `mean` (mean-pool, 512-d), `both` (concat, 1024-d) |

### Projection head

```
proj_input → Linear(in, 512) → ReLU → Linear(512, 128) → L2-normalise
```

### SupCon loss (Khosla L_sup_out)

For each anchor *i* in a batch of *B* samples:
- **P(i)** = same-class samples excluding self — positives
- **A(i)** = all samples excluding self — denominator

```
L = (1/|B'|) Σ_{i ∈ B'}  −(1/|P(i)|) Σ_{p ∈ P(i)}  log[ exp(z_i·z_p/τ) / Σ_{a ∈ A(i)} exp(z_i·z_a/τ) ]
```

where `B'` is the set of anchors with at least one positive (singleton-class anchors are excluded from the mean). Computed in fp32 with the log-sum-exp trick for numerical safety under AMP.

---

## LoRA configuration

- **Targets**: `transformer.layers.{i}.0.to_qkv` and `transformer.layers.{i}.0.to_out` (22 layers × 2 = 44 modules)
- **Default**: rank=16, alpha=16, dropout=0.0 (scaling factor α/r = 1.0)
- **Library**: peft (`LoraConfig` + `get_peft_model`)
- Injection utility reused from `src/approaches/fine_tuning/lora.py`

---

## Saved checkpoints

Checkpoints live under `outputs/jade_checkpoints/<run_name>/`. Only the best fold (highest val_acc) is retained on disk; the others are deleted as training progresses.

- **LoRA mode**: `lora_adapter/` (peft convention) + `head_weights.pt` (cls_query_token + linear_head) + `projection_head.pt`
- **Full FT mode**: `full_model.pt` (complete state dict, projection head excluded) + `projection_head.pt`
- **Fold metadata**: `fold_meta.json` (val subjects, val metrics, SupCon hyperparameters)
- **LP warmup best state** is held in memory and loaded with `strict=False` before stage 2 begins

The projection head is always saved separately for downstream analysis (embedding visualisation, cross-run comparison). It is not required for classification inference — only `linear_head` + (encoder | encoder+LoRA) are needed.

---

## Outputs

- Checkpoints: `outputs/jade_checkpoints/<run_name>/`
- Summary JSON: `outputs/jade_checkpoints/summary_*.json`
- W&B project: `eeg-jade-v2` (set `USE_WANDB = False` in `config.py` to disable)
- Metrics: accuracy, balanced accuracy, AUROC (macro OvR for 9-class), weighted F1
- Extra per-epoch W&B metrics during FT: `train/loss_ce`, `train/loss_sc` (component losses)

---

## Submitting experiments on DelftBlue

A first-wave batch script is provided at `slurm/run_jade_experiments.sh`. It
submits 12 FACED jobs (baseline LoRA + SupCon, α sweep, full-FT probe) covering
both CV and stimulus-generalisation evaluation.

```bash
bash slurm/run_jade_experiments.sh
squeue --me                    # monitor queue
tail -f slurm/logs/<jobid>.out # follow a running job
```
