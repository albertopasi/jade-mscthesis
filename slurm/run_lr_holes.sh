#!/bin/bash
# Fill missing cells in Stage 2b — 9-class single-fold LR sweep.
# (See docs/jade_hp_methodology.md §4.2b)
#
# Missing: lr=5e-5 (previous run failed), lr=2e-4 (never run).
# Both at fold 1, B=256, α=0.3, τ=0.1.
#
# Usage: bash slurm/run_lr_holes.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

MODULE="src.approaches.jade.train_jade"
COMMON="--dataset faced --task 9-class --fullft \
        --alpha 0.3 --temperature 0.1 \
        --batch-size 256 --fold 1"

echo "=== Stage 2b LR-sweep holes (9-class fold 1) ==="

for LR in 2e-4; do
    JOB=$(sbatch --job-name="jade-9cl-lr${LR}" \
                 --time=02:00:00 \
                 slurm/run_experiment.sh $MODULE $COMMON --ft-lr $LR)
    echo "  lr=$LR  $JOB"
done

echo ""
echo "Logs in slurm/logs/. Add results to docs/jade_hp_methodology.md 4.2b."
