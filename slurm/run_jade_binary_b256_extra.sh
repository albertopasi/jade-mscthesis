#!/bin/bash
# JADE binary @ B=256, lr=1e-4 — extra (α, τ) grid.
# Investigates whether sharper τ or slightly higher α can push binary past B=128.
#
# Grid: α ∈ {0.2, 0.3} × τ ∈ {0.03, 0.1} = 4 jobs × 10-fold CV.
# Walltime: 10h/job.
#
# Usage: bash slurm/run_jade_binary_b256_extra.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

MODULE="src.approaches.jade.train_jade"
BASE="--dataset faced --task binary --fullft --batch-size 256 --ft-lr 1e-4"

echo "=== JADE binary @ B=256 lr=1e-4 extra grid (4 jobs) ==="
echo ""

for ALPHA in 0.2 0.3; do
    for TAU in 0.03 0.1; do
        JOB=$(sbatch --job-name="jade-bin-a${ALPHA}-t${TAU}" \
                     --time=10:00:00 \
                     slurm/run_experiment.sh $MODULE $BASE \
                     --alpha $ALPHA --temperature $TAU)
        echo "  a=$ALPHA t=$TAU  $JOB"
    done
done

echo ""
echo "All 4 jobs submitted. Monitor: squeue -u \$USER"
