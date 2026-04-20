#!/bin/bash
# Additional τ sweep: τ=0.03 (sharper than the 0.05 tested in run_tau_sweep.sh).
#
# Fixed: fullft, pooling=no, repr=context, α ∈ {0.2, 0.3}, τ = 0.03
#
# 4 jobs: 1 τ × 2 α × 2 tasks
#
# Usage:
#   bash slurm/run_tau_sweep_extra.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

MODULE="src.approaches.jade.train_jade"

echo "=== Submitting JADE τ=0.03 sweep (FullFT, FACED, α∈{0.2,0.3}) ==="
echo ""

JOB=$(sbatch --job-name="jade-bin-a02-t003" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task binary --fullft --alpha 0.2 --temperature 0.03)
echo " 1/4 faced binary  fullft a=0.2 t=0.03: $JOB"

JOB=$(sbatch --job-name="jade-bin-a03-t003" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task binary --fullft --alpha 0.3 --temperature 0.03)
echo " 2/4 faced binary  fullft a=0.3 t=0.03: $JOB"

JOB=$(sbatch --job-name="jade-9cl-a02-t003" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task 9-class --fullft --alpha 0.2 --temperature 0.03)
echo " 3/4 faced 9-class fullft a=0.2 t=0.03: $JOB"

JOB=$(sbatch --job-name="jade-9cl-a03-t003" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task 9-class --fullft --alpha 0.3 --temperature 0.03)
echo " 4/4 faced 9-class fullft a=0.3 t=0.03: $JOB"

echo ""
echo "All 4 τ=0.03 jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs will be in: slurm/logs/"
