#!/bin/bash
# Verify whether lr=2e-4 beats lr=4e-4 at full 10-fold CV.
# (Fold-1 lr=2e-4 was 0.6703 vs lr=4e-4 0.6593 — within run-to-run noise;
#  this run gives the definitive comparison.)
#
# Config: 9-class fullft, α=0.3, τ=0.2, B=256, lr=2e-4.
# Compares directly to current best (62.61 at lr=4e-4, same α/τ).
#
# Usage: bash slurm/run_lr_holes.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

JOB=$(sbatch --job-name="jade-9cl-lr2e-4-cv" \
             --time=10:00:00 \
             slurm/run_experiment.sh src.approaches.jade.train_jade \
             --dataset faced --task 9-class --fullft \
             --alpha 0.3 --temperature 0.2 \
             --batch-size 256 --ft-lr 2e-4)
echo "  jade-9cl-lr2e-4-cv  $JOB"
echo ""
echo "On completion, compare to 62.61 (current best @ lr=4e-4)."
