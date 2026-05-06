#!/bin/bash
# FACED gen seed=789 for FT-FullFT and JADE-FullFT.
# Completes the 3-seed coverage for the headline configs (LP s789 already exists).
#
# Configs (matched to bulletproof CV winners):
#   FT-FullFT  9-class  @ B=256, lr=4e-4, no-mixup
#   FT-FullFT  binary   @ B=128, lr=1e-4, no-mixup
#   JADE-FullFT 9-class @ α=0.3, τ=0.2,  B=256, lr=4e-4
#   JADE-FullFT binary  @ α=0.2, τ=0.05, B=128, lr=1e-4
#
# 4 jobs × 1 seed × 10 folds × 10h walltime each.
#
# Usage: bash slurm/run_faced_gen_s789.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

JADE="src.approaches.jade.train_jade"
FT="src.approaches.fine_tuning.train_ft"

echo "=== FACED gen seed=789 (4 jobs) ==="
echo ""

echo "FT-FullFT:"
JOB=$(sbatch --job-name="ft-gen-9cl-s789" --time=10:00:00 \
             slurm/run_experiment.sh $FT \
             --dataset faced --task 9-class --fullft --no-mixup --generalization \
             --gen-seeds 789 \
             --batch-size 256 --ft-lr 4e-4)
echo "  ft-gen-9cl-s789  $JOB"
JOB=$(sbatch --job-name="ft-gen-bin-s789" --time=10:00:00 \
             slurm/run_experiment.sh $FT \
             --dataset faced --task binary --fullft --no-mixup --generalization \
             --gen-seeds 789 \
             --batch-size 128 --ft-lr 1e-4)
echo "  ft-gen-bin-s789  $JOB"

echo ""
echo "JADE-FullFT:"
JOB=$(sbatch --job-name="jade-gen-9cl-s789" --time=10:00:00 \
             slurm/run_experiment.sh $JADE \
             --dataset faced --task 9-class --fullft --generalization \
             --gen-seeds 789 \
             --alpha 0.3 --temperature 0.2 \
             --batch-size 256 --ft-lr 4e-4)
echo "  jade-gen-9cl-s789  $JOB"
JOB=$(sbatch --job-name="jade-gen-bin-s789" --time=10:00:00 \
             slurm/run_experiment.sh $JADE \
             --dataset faced --task binary --fullft --generalization \
             --gen-seeds 789 \
             --alpha 0.2 --temperature 0.05 \
             --batch-size 128 --ft-lr 1e-4)
echo "  jade-gen-bin-s789  $JOB"

echo ""
echo "All 4 jobs submitted. Monitor: squeue -u \$USER"
echo "On completion: 3-seed gen coverage complete for LP / FT / JADE × {9-class, binary}."
