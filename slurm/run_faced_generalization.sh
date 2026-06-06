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

echo "=== FACED gen — LP @ seeds 123 789 (1 job) ==="
echo ""

JOB=$(sbatch --job-name="lp-gen-bin" --time=04:30:00 \
             slurm/run_experiment.sh $LP \
             --dataset faced --task binary --no-mixup --generalization \
             --gen-seeds 123 789)
echo "  lp-gen-bin  $JOB"

echo ""
echo "Job submitted. Monitor: squeue -me -start"
