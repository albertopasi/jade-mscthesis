#!/bin/bash
# FACED generalization — LP coverage + 1 additional seed for FT & JADE.
#
# Adds:
#   - LP @ official recipe, both tasks, seed=123  (matches the FT/JADE protocol;
#     prior LP gen runs were at older defaults and not directly comparable)
#   - Seed 456 for the 4 already-completed FT and JADE configs (so we have a
#     2-seed average to check the seed=123 result is not idiosyncratic)
#
# 6 jobs × 1 seed × 10 folds × 10h walltime each.
#
# Usage: bash slurm/run_faced_gen_extra.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

LP="src.approaches.linear_probing.train_lp"
JADE="src.approaches.jade.train_jade"
FT="src.approaches.fine_tuning.train_ft"

echo "=== FACED gen — LP + extra seed (6 jobs) ==="
echo ""

echo "LP (official, frozen encoder), seed=123:"
JOB=$(sbatch --job-name="lp-gen-9cl-s123" --time=7:00:00 \
             slurm/run_experiment.sh $LP \
             --dataset faced --task 9-class --generalization \
             --gen-seeds 123)
echo "  lp-gen-9cl-s123  $JOB"
JOB=$(sbatch --job-name="lp-gen-bin-s123" --time=7:00:00 \
             slurm/run_experiment.sh $LP \
             --dataset faced --task binary --generalization \
             --gen-seeds 123)
echo "  lp-gen-bin-s123  $JOB"

echo ""
echo "FT-FullFT extra seed=456:"
JOB=$(sbatch --job-name="ft-gen-9cl-s456" --time=10:00:00 \
             slurm/run_experiment.sh $FT \
             --dataset faced --task 9-class --fullft --no-mixup --generalization \
             --gen-seeds 456 \
             --batch-size 256 --ft-lr 4e-4)
echo "  ft-gen-9cl-s456  $JOB"
JOB=$(sbatch --job-name="ft-gen-bin-s456" --time=10:00:00 \
             slurm/run_experiment.sh $FT \
             --dataset faced --task binary --fullft --no-mixup --generalization \
             --gen-seeds 456 \
             --batch-size 128 --ft-lr 1e-4)
echo "  ft-gen-bin-s456  $JOB"

echo ""
echo "JADE-FullFT extra seed=456:"
JOB=$(sbatch --job-name="jade-gen-9cl-s456" --time=10:00:00 \
             slurm/run_experiment.sh $JADE \
             --dataset faced --task 9-class --fullft --generalization \
             --gen-seeds 456 \
             --alpha 0.3 --temperature 0.2 \
             --batch-size 256 --ft-lr 4e-4)
echo "  jade-gen-9cl-s456  $JOB"
JOB=$(sbatch --job-name="jade-gen-bin-s456" --time=10:00:00 \
             slurm/run_experiment.sh $JADE \
             --dataset faced --task binary --fullft --generalization \
             --gen-seeds 456 \
             --alpha 0.2 --temperature 0.05 \
             --batch-size 128 --ft-lr 1e-4)
echo "  jade-gen-bin-s456  $JOB"

echo ""
echo "All 6 jobs submitted. Monitor: squeue -u \$USER"
echo "Add seed=789 later if cross-seed averaging needs more confidence."
