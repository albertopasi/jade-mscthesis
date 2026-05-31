#!/bin/bash
# FACED stimulus-generalization — fill in FT and LP for seeds 123 + 789.
# JADE is already done at these seeds; only FT and LP missing.
#
# 4 jobs total — each runs both seeds in one job:
#   FT  9-class @ B=256, lr=4e-4 (20h)
#   FT  binary  @ B=128, lr=1e-4 (20h)
#   LP  9-class @ official       ( 6h)
#   LP  binary  @ official       ( 6h)
#
# Usage: bash slurm/run_faced_generalization.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

FT="src.approaches.fine_tuning.train_ft"
LP="src.approaches.linear_probing.train_lp"

echo "=== FACED gen — FT + LP @ seeds 123 789 (4 jobs) ==="
echo ""

echo "FT-FullFT (20h each):"
JOB=$(sbatch --job-name="ft-gen-9cl" --time=20:00:00 \
             slurm/run_experiment.sh $FT \
             --dataset faced --task 9-class --fullft --no-mixup --generalization \
             --gen-seeds 123 789 --no-save-checkpoints \
             --batch-size 256 --ft-lr 4e-4)
echo "  ft-gen-9cl  $JOB"

JOB=$(sbatch --job-name="ft-gen-bin" --time=20:00:00 \
             slurm/run_experiment.sh $FT \
             --dataset faced --task binary --fullft --no-mixup --generalization \
             --gen-seeds 123 789 --no-save-checkpoints \
             --batch-size 128 --ft-lr 1e-4)
echo "  ft-gen-bin  $JOB"

echo ""
echo "LP (no-mixup for parity with FT/JADE, 6h each):"
JOB=$(sbatch --job-name="lp-gen-9cl" --time=06:00:00 \
             slurm/run_experiment.sh $LP \
             --dataset faced --task 9-class --no-mixup --generalization \
             --gen-seeds 123 789)
echo "  lp-gen-9cl  $JOB"

JOB=$(sbatch --job-name="lp-gen-bin" --time=06:00:00 \
             slurm/run_experiment.sh $LP \
             --dataset faced --task binary --no-mixup --generalization \
             --gen-seeds 123 789)
echo "  lp-gen-bin  $JOB"

echo ""
echo "All 4 jobs submitted. Monitor: squeue -u \$USER"
