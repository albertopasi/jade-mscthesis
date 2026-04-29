#!/bin/bash
# JADE-FullFT Stage 3 grid completion for FACED.
# Fills the diagonal cells left blank by the bulletproof plus-shape sweep,
# producing complete 3×4 (9-class) and 3×5 (binary) grids around the
# Stage 3 winners. None of these are expected to change the winner;
# they exist to make the thesis tables hole-free.
#
# 9-class (B=256, lr=4e-4) — 5 cells:
#   α=0.2 × τ ∈ {0.05, 0.5}
#   α=0.5 × τ ∈ {0.05, 0.1, 0.5}
#
# Binary (B=256, lr=1e-4) — 6 cells:
#   α=0.2 × τ ∈ {0.2, 0.5}
#   α=0.5 × τ ∈ {0.05, 0.1, 0.2, 0.5}
#
# Binary low-LR sanity checks at both batch sizes — 2 cells:
#   B=128, α=0.2, τ=0.05, lr=5e-5  (verify B=128 lr=1e-4 isn't dominated by smaller LR)
#   B=256, α=0.3, τ=0.03, lr=5e-5  (test if smaller LR rescues B=256 binary)
#
# 13 jobs × 10-fold CV × 10h walltime each.
#
# Usage: bash slurm/run_jade_grid_completion.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

JADE="src.approaches.jade.train_jade"

submit() {
    local task=$1 alpha=$2 tau=$3 bs=$4 lr=$5 label=$6
    local short="${task/9-class/9cl}"; short="${short/binary/bin}"
    local name="jade-${short}-a${alpha}-t${tau}-b${bs}-lr${lr}"
    JOB=$(sbatch --job-name="$name" --time=10:00:00 \
                 slurm/run_experiment.sh $JADE \
                 --dataset faced --task $task --fullft \
                 --alpha $alpha --temperature $tau \
                 --batch-size $bs --ft-lr $lr)
    echo "  [$label] $name  $JOB"
}

echo "=== JADE grid-completion sweep — 12 jobs ==="
echo ""

echo "9-class @ B=256, lr=4e-4 — diagonal cells:"
submit 9-class 0.2 0.05 256 4e-4 "9cl-α0.2"
submit 9-class 0.2 0.5  256 4e-4 "9cl-α0.2"
submit 9-class 0.5 0.05 256 4e-4 "9cl-α0.5"
submit 9-class 0.5 0.1  256 4e-4 "9cl-α0.5"
submit 9-class 0.5 0.5  256 4e-4 "9cl-α0.5"

echo ""
echo "Binary @ B=256, lr=1e-4 — diagonal cells:"
submit binary  0.2 0.2  256 1e-4 "bin-α0.2"
submit binary  0.2 0.5  256 1e-4 "bin-α0.2"
submit binary  0.5 0.05 256 1e-4 "bin-α0.5"
submit binary  0.5 0.1  256 1e-4 "bin-α0.5"
submit binary  0.5 0.2  256 1e-4 "bin-α0.5"
submit binary  0.5 0.5  256 1e-4 "bin-α0.5"

echo ""
echo "Binary low-LR sanity checks (each batch's winner config at lr=5e-5):"
submit binary  0.2 0.05 128 5e-5 "bin-B128-LR↓"
submit binary  0.3 0.03 256 5e-5 "bin-B256-LR↓"

echo ""
echo "All 13 jobs submitted. Monitor: squeue -u \$USER"
echo "Expected wall-clock: ~3-4 days at 2-concurrent limit."
