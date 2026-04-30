#!/bin/bash
# FACED stimulus-generalization at the bulletproof-CV winners.
#
# Each job: 1 seed × 10 folds × 10h walltime.
# HPs and recipe are the cross-subject-CV optima, applied unchanged
# (standard train/val/test methodology — no re-tuning on gen).
#
# Configs:
#   9-class JADE-FullFT @ α=0.3, τ=0.2, B=256, lr=4e-4
#   9-class FT-FullFT   @ B=256, lr=4e-4, no-mixup    (matched baseline)
#   binary  JADE-FullFT @ α=0.2, τ=0.05, B=128, lr=1e-4
#   binary  FT-FullFT   @ B=128, lr=1e-4, no-mixup    (matched baseline)
#
# Default seed: 123. Add seeds {456, 789} later if the 1-seed result
# motivates cross-seed averaging.
#
# Usage: bash slurm/run_faced_generalization.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

JADE="src.approaches.jade.train_jade"
FT="src.approaches.fine_tuning.train_ft"

echo "=== FACED generalization (1 seed × 4 jobs) ==="
echo ""

echo "9-class @ B=256, lr=4e-4:"
JOB=$(sbatch --job-name="jade-gen-9cl" --time=10:00:00 \
             slurm/run_experiment.sh $JADE \
             --dataset faced --task 9-class --fullft --generalization \
             --gen-seeds 123 \
             --alpha 0.3 --temperature 0.2 \
             --batch-size 256 --ft-lr 4e-4)
echo "  jade-gen-9cl  $JOB"
JOB=$(sbatch --job-name="ft-gen-9cl" --time=10:00:00 \
             slurm/run_experiment.sh $FT \
             --dataset faced --task 9-class --fullft --no-mixup --generalization \
             --gen-seeds 123 \
             --batch-size 256 --ft-lr 4e-4)
echo "  ft-gen-9cl    $JOB"

echo ""
echo "binary @ B=128, lr=1e-4:"
JOB=$(sbatch --job-name="jade-gen-bin" --time=10:00:00 \
             slurm/run_experiment.sh $JADE \
             --dataset faced --task binary --fullft --generalization \
             --gen-seeds 123 \
             --alpha 0.2 --temperature 0.05 \
             --batch-size 128 --ft-lr 1e-4)
echo "  jade-gen-bin  $JOB"
JOB=$(sbatch --job-name="ft-gen-bin" --time=10:00:00 \
             slurm/run_experiment.sh $FT \
             --dataset faced --task binary --fullft --no-mixup --generalization \
             --gen-seeds 123 \
             --batch-size 128 --ft-lr 1e-4)
echo "  ft-gen-bin    $JOB"

echo ""
echo "All 4 jobs submitted. Monitor: squeue -u \$USER"
echo "Seed: 123. Add 456 / 789 later if cross-seed averaging is needed."
