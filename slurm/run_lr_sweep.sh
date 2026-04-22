#!/bin/bash
# Single-fold LR sweep at B=256, FACED 9-class fullft, α=0.3, τ=0.1.
# Full ft-epochs so LR comparison is meaningful (not a smoke test).
#
# Usage: bash slurm/run_lr_sweep.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

MODULE="src.approaches.jade.train_jade"
COMMON="--dataset faced --task 9-class --fullft \
        --alpha 0.3 --temperature 0.1 \
        --batch-size 256 --fold 1"

echo "=== LR sweep @ B=256 (FACED 9-class fullft, fold 1) ==="

for LR in 8e-4 1.5e-3; do
    JOB=$(sbatch --job-name="jade-lr${LR}" \
                 --time=01:00:00 \
                 slurm/run_experiment.sh $MODULE $COMMON --ft-lr $LR)
    echo "  lr=$LR  $JOB"
done

echo ""
echo "Logs in slurm/logs/. Compare best val_acc across the 4 runs."
