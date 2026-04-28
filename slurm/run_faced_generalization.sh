#!/bin/bash
# FACED stimulus-generalization tests for JADE-best vs FT-FullFT.
#
# Each job runs 3 seeds × 10 folds = 30 fold-runs. Walltime 30h.
# Configs match the cross-subject CV bests (per-task batch + LR).
#
# 9-class (lr=4e-4):
#   JADE α=0.3 τ=0.2 B=256  vs  FT B=256
# binary:
#   JADE α=0.3 τ=0.03 B=256 lr=1e-4  vs  FT B=256 lr=2e-4
#
# Usage: bash slurm/run_faced_generalization.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

JADE="src.approaches.jade.train_jade"
FT="src.approaches.fine_tuning.train_ft"

echo "=== FACED generalization (4 jobs × 3 seeds × 10 folds) ==="
echo ""

echo "9-class:"
JOB=$(sbatch --job-name="jade-gen-9cl" --time=30:00:00 \
             slurm/run_experiment.sh $JADE \
             --dataset faced --task 9-class --fullft --generalization \
             --alpha 0.3 --temperature 0.2 \
             --batch-size 256 --ft-lr 4e-4)
echo "  jade-gen-9cl  $JOB"
JOB=$(sbatch --job-name="ft-gen-9cl" --time=30:00:00 \
             slurm/run_experiment.sh $FT \
             --dataset faced --task 9-class --fullft --no-mixup --generalization \
             --batch-size 256 --ft-lr 4e-4)
echo "  ft-gen-9cl    $JOB"

echo ""
echo "binary:"
JOB=$(sbatch --job-name="jade-gen-bin" --time=30:00:00 \
             slurm/run_experiment.sh $JADE \
             --dataset faced --task binary --fullft --generalization \
             --alpha 0.3 --temperature 0.03 \
             --batch-size 256 --ft-lr 1e-4)
echo "  jade-gen-bin  $JOB"
JOB=$(sbatch --job-name="ft-gen-bin" --time=30:00:00 \
             slurm/run_experiment.sh $FT \
             --dataset faced --task binary --fullft --no-mixup --generalization \
             --batch-size 256 --ft-lr 2e-4)
echo "  ft-gen-bin    $JOB"

echo ""
echo "All 4 jobs submitted. Monitor: squeue -u \$USER"
echo "Default seeds: 123 456 789"
