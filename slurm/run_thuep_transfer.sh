#!/bin/bash
# THU-EP direct transfer of FACED-optimal configs.
#
# JADE uses task-dependent recipes (batch/LR differ between tasks) that were
# tuned on FACED. This script applies them to THU-EP without re-sweeping, so
# the result speaks to transferability. Each JADE run is paired with a
# matching FT-FullFT baseline for attribution.
#
# Grid (4 jobs × 10-fold CV):
#   9-class: JADE α=0.3 τ=0.2 B=256 lr=4e-4  vs  FT B=256 lr=4e-4
#   binary:  JADE α=0.3 τ=0.03 B=256 lr=1e-4 vs  FT B=256 lr=2e-4
#
# Walltime: 10h/job.
#
# Usage: bash slurm/run_thuep_transfer.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

JADE="src.approaches.jade.train_jade"
FT="src.approaches.fine_tuning.train_ft"

echo "=== THU-EP transfer (4 jobs) ==="
echo ""

echo "9-class (lr=4e-4):"
JOB=$(sbatch --job-name="jade-thuep-9cl" --time=10:00:00 \
             slurm/run_experiment.sh $JADE \
             --dataset thu-ep --task 9-class --fullft \
             --alpha 0.3 --temperature 0.2 \
             --batch-size 256 --ft-lr 4e-4)
echo "  jade-thuep-9cl  $JOB"
JOB=$(sbatch --job-name="ft-thuep-9cl" --time=10:00:00 \
             slurm/run_experiment.sh $FT \
             --dataset thu-ep --task 9-class --fullft --no-mixup \
             --batch-size 256 --ft-lr 4e-4)
echo "  ft-thuep-9cl    $JOB"

echo ""
echo "binary:"
JOB=$(sbatch --job-name="jade-thuep-bin" --time=10:00:00 \
             slurm/run_experiment.sh $JADE \
             --dataset thu-ep --task binary --fullft \
             --alpha 0.3 --temperature 0.03 \
             --batch-size 256 --ft-lr 1e-4)
echo "  jade-thuep-bin  $JOB"
JOB=$(sbatch --job-name="ft-thuep-bin" --time=10:00:00 \
             slurm/run_experiment.sh $FT \
             --dataset thu-ep --task binary --fullft --no-mixup \
             --batch-size 256 --ft-lr 2e-4)
echo "  ft-thuep-bin    $JOB"

echo ""
echo "All 4 jobs submitted. Monitor: squeue -u \$USER"
