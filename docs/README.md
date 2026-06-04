# `docs/` — Annotated Index

This folder holds the long-form documentation for the thesis. The repo's [`README.md`](../README.md) is the entry point and orientation; the files here go deeper.

Use this index as your map. Each entry lists the question the doc answers, so you can jump straight to the right file.

---

## 🧭 Start here

| Document | Read this when you want to know… |
|---|---|
| [`results_brief.md`](results_brief.md) | …the detailed numbers — headline accuracies, per-class metrics, confusion matrices, subject-wise breakdowns, paired wins/losses. Extracted directly from `main-results/`. The closest thing to a self-contained companion to the thesis Results chapter. |

---

## 🔬 Methodology

| Document | Read this when you want to know… |
|---|---|
| [`jade_approach_design.md`](jade_approach_design.md) | …why JADE is designed the way it is — joint loss formulation, projection-head architecture, two-stage schedule rationale. |
| [`jade_hp_sweep.md`](jade_hp_sweep.md) | …how the JADE hyperparameters were chosen — the staged sweep protocol (loss HPs → optimization → re-verification), the full sweep tables behind each decision, and the per-task final configuration with its rationale. |
| [`architecture.md`](architecture.md) | …how the training and inference pipelines are wired together — config conventions, run-name patterns, the `FoldPredictions` + `write_run_summary` design, REVE integration details. |
| [`reproducing_results.md`](reproducing_results.md) | …the exact command for any specific number, table, or figure in the thesis. |

---

## 📊 Results

| Document | Read this when you want to know… |
|---|---|
| [`statistical_tests.md`](statistical_tests.md) | …whether a result is significant — paired Wilcoxon, paired t, sign test, BCa CIs, Holm correction. **Auto-generated** by `uv run python -m src.inference.statistical_tests`. |
| [`statistical_tests_generalization.md`](statistical_tests_generalization.md) | …whether the methods differ on the stimulus-generalization protocol — Friedman omnibus, Brown-Forsythe-via-Friedman dispersion test, variance-ratio CIs. **Auto-generated** by `uv run python -m src.inference.statistical_tests_generalization`. |
| [`lp_results_analysis.md`](lp_results_analysis.md) | …the LP-specific ablations (pooling mode, window size, mixup) and the AUROC–accuracy dissociation observation that motivates fine-tuning. |

---

## 🗂️ Bookkeeping

| Document | Read this when you want to know… |
|---|---|
| [`runs_inventory.md`](runs_inventory.md) | …which runs exist, which sweep cell each one belongs to, and which gaps in the matrix are still missing. |

---

## ⚙️ Conventions

- **Auto-generated docs** carry a header that says so. Don't hand-edit them — re-run the script.
- **All numbers** are extracted from `main-results/<approach>_<task>/<stem>.json` so you can always trace a value back to the run that produced it.
- **All paths** in these docs are relative to the repo root.
- **Date/version stamps** are not maintained — git history is the source of truth.
