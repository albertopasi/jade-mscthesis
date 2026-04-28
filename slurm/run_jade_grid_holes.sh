#!/bin/bash
# Fill-in jobs for hole-free Stage 1 / Stage 3 grids.
# See docs/jade_hp_methodology.md §8 for context.
#
# Stage 1 (B=128, lr=1e-4 — REVE default optimization):
#   1. binary α=0.3 τ=0.5    — complete main grid
#   2. binary α=0.9 τ=0.1    — symmetric α-extreme ablation
#
# Stage 3 (B=256, tuned lr — re-grid around Stage 1 winner):
#   3. 9-class α=0.2 τ=0.1 lr=4e-4
#   4. 9-class α=0.2 τ=0.2 lr=4e-4
#   5. binary  α=0.3 τ=0.05 lr=1e-4
#
# 5 jobs × 10-fold CV. Walltime 10h each.
#
# Usage: bash slurm/run_jade_grid_holes.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

JADE="src.approaches.jade.train_jade"

submit() {
    local task=$1 alpha=$2 tau=$3 bs=$4 lr=$5 stage=$6
    local short="${task/9-class/9cl}"; short="${short/binary/bin}"
    local name="jade-${short}-a${alpha}-t${tau}-b${bs}-lr${lr}"
    JOB=$(sbatch --job-name="$name" --time=10:00:00 \
                 slurm/run_experiment.sh $JADE \
                 --dataset faced --task $task --fullft \
                 --alpha $alpha --temperature $tau \
                 --batch-size $bs --ft-lr $lr)
    echo "  [Stage $stage] $name  $JOB"
}

echo "=== JADE grid-hole fillers (5 jobs) ==="
echo ""

echo "Stage 1 holes (B=128, lr=1e-4):"
submit binary  0.3 0.5 128 1e-4 1
submit binary  0.9 0.1 128 1e-4 1

echo ""
echo "Stage 3 holes (B=256, tuned lr):"
submit 9-class 0.2 0.1 256 4e-4 3
submit 9-class 0.2 0.2 256 4e-4 3
submit binary  0.3 0.05 256 1e-4 3

echo ""
echo "All 5 jobs submitted. Monitor: squeue -u \$USER"
