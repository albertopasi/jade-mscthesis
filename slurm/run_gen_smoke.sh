#!/bin/bash
# Smoke test for the new generalization + in-training subject-wise inference path.
#
# Each job runs one fold with very few epochs, exercising:
#   1. The new run_fold_inference() call after best-epoch restore.
#   2. The new write_run_summary() call at end-of-seed in main().
#   3. The --no-save-checkpoints flag (JADE / FT only — LP keeps checkpoints).
#
# Success criteria (per job):
#   - Job finishes with exit code 0.
#   - main-results/<approach>_<task>_generalization/<stem>_gen_s123.json exists
#     with a non-empty `per_subject_acc` dict.
#   - Matching .npz exists alongside.
#   - For JADE/FT: no per-fold checkpoint dir under outputs/<approach>_checkpoints/.
#   - For LP: classifier_weights.pt present under outputs/lp_checkpoints/ (existing behavior).
#
# Usage: bash slurm/run_gen_smoke.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

LP="src.approaches.linear_probing.train_lp"
FT="src.approaches.fine_tuning.train_ft"
JADE="src.approaches.jade.train_jade"

echo "=== Generalization-path smoke test (3 jobs, each fold 1, tiny epochs) ==="
echo ""

# # ── LP smoke: official mode, fold 1, single seed ─────────────────────────────
# JOB=$(sbatch --job-name="smoke-lp" --time=00:10:00 \
#              slurm/run_experiment.sh $LP \
#              --dataset faced --task 9-class --generalization \
#              --gen-seeds 123 \
#              --fold 1 --epochs 3)
# echo "  smoke-lp     $JOB"

# # ── FT smoke: full FT, fold 1, no-mixup, no checkpoints ──────────────────────
# JOB=$(sbatch --job-name="smoke-ft" --time=00:10:00 \
#              slurm/run_experiment.sh $FT \
#              --dataset faced --task 9-class --fullft --no-mixup --generalization \
#              --gen-seeds 123 --no-save-checkpoints \
#              --fold 1 --lp-epochs 2 --ft-epochs 3 \
#              --batch-size 256 --ft-lr 4e-4)
# echo "  smoke-ft     $JOB"

# ── JADE smoke: full FT, fold 1, no checkpoints ──────────────────────────────
JOB=$(sbatch --job-name="jade9class" --time=18:00:00 \
             slurm/run_experiment.sh $JADE \
             --dataset faced --task 9-class --fullft --generalization \
             --gen-seeds 123 789 --no-save-checkpoints \
             --alpha 0.3 --temperature 0.2 \
             --batch-size 256 --ft-lr 4e-4)

JOB=$(sbatch --job-name="jadebin" --time=18:00:00 \
             slurm/run_experiment.sh $JADE \
             --dataset faced --task binary --fullft --generalization \
             --gen-seeds 123 789 --no-save-checkpoints \
             --alpha 0.2 --temperature 0.05 \
             --batch-size 128 --ft-lr 1e-4)
echo "  smoke-jade   $JOB"


