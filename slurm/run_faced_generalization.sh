#!/bin/bash
# FACED stimulus-generalization at the bulletproof-CV winners.
#
# Each job: 3 seeds × 10 folds (3 gen seeds = three independent stimulus splits).
# Per-seed subject-wise summaries are written by the training script under
# main-results/<approach>_<task>_generalization/ as part of each fold's
# end-of-training inference pass — no per-fold checkpoints are saved
# (--no-save-checkpoints).
#
# After all jobs finish, average across seeds with:
#   uv run python -m src.inference.average_gen_seeds --approach jade --task 9-class
#
# NOTE: walltime below was set for 1 seed. Bump it to ~30h for 3 seeds, or
# split into per-seed jobs (one --gen-seeds value each) to parallelize.
#
# Configs:
#   9-class JADE-FullFT @ α=0.3, τ=0.2, B=256, lr=4e-4
#   9-class FT-FullFT   @ B=256, lr=4e-4, no-mixup    (matched baseline)
#   binary  JADE-FullFT @ α=0.2, τ=0.05, B=128, lr=1e-4
#   binary  FT-FullFT   @ B=128, lr=1e-4, no-mixup    (matched baseline)
#
# Usage: bash slurm/run_faced_generalization.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

JADE="src.approaches.jade.train_jade"
FT="src.approaches.fine_tuning.train_ft"

echo "=== FACED generalization (3 seeds × 4 jobs) ==="
echo ""

echo "9-class @ B=256, lr=4e-4:"
JOB=$(sbatch --job-name="jade-gen-9cl" --time=10:00:00 \
             slurm/run_experiment.sh $JADE \
             --dataset faced --task 9-class --fullft --generalization \
             --gen-seeds 123 456 789 --no-save-checkpoints \
             --alpha 0.3 --temperature 0.2 \
             --batch-size 256 --ft-lr 4e-4)
echo "  jade-gen-9cl  $JOB"
JOB=$(sbatch --job-name="ft-gen-9cl" --time=10:00:00 \
             slurm/run_experiment.sh $FT \
             --dataset faced --task 9-class --fullft --no-mixup --generalization \
             --gen-seeds 123 456 789 --no-save-checkpoints \
             --batch-size 256 --ft-lr 4e-4)
echo "  ft-gen-9cl    $JOB"

echo ""
echo "binary @ B=128, lr=1e-4:"
JOB=$(sbatch --job-name="jade-gen-bin" --time=10:00:00 \
             slurm/run_experiment.sh $JADE \
             --dataset faced --task binary --fullft --generalization \
             --gen-seeds 123 456 789 --no-save-checkpoints \
             --alpha 0.2 --temperature 0.05 \
             --batch-size 128 --ft-lr 1e-4)
echo "  jade-gen-bin  $JOB"
JOB=$(sbatch --job-name="ft-gen-bin" --time=10:00:00 \
             slurm/run_experiment.sh $FT \
             --dataset faced --task binary --fullft --no-mixup --generalization \
             --gen-seeds 123 456 789 --no-save-checkpoints \
             --batch-size 128 --ft-lr 1e-4)
echo "  ft-gen-bin    $JOB"

echo ""
echo "All 4 jobs submitted. Monitor: squeue -u \$USER"
echo "Seeds: 123 456 789. Each job emits 3 per-seed JSONs in main-results/<approach>_<task>_generalization/."
