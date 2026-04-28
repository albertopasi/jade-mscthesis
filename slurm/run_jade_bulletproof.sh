#!/bin/bash
# Bulletproof JADE-FullFT HP sweep on FACED.
# Submits the 12 cells needed to make the (α, τ, lr) sweep methodology
# unattackable. See docs/jade_hp_methodology.md.
#
# Composition:
#   Stage 1 holes (B=128, lr=1e-4):
#     1. binary α=0.3 τ=0.5    — main-grid hole
#     2. binary α=0.9 τ=0.1    — α-ablation symmetry
#
#   Stage 3 — 9-class (B=256, lr=4e-4) plus-shape around winner (α=0.3, τ=0.2):
#     3. α=0.2 τ=0.1           — α-axis
#     4. α=0.2 τ=0.2           — α-axis at winning τ
#     5. α=0.5 τ=0.2           — α-axis at winning τ (high side)
#     6. α=0.3 τ=0.05          — τ-axis at winning α
#     7. α=0.3 τ=0.5           — τ-axis at winning α (high side)
#
#   Stage 3 — 9-class LR cross-check at winner (B=256):
#     8. α=0.3 τ=0.2 lr=1e-4   — triangulate LR axis at the winning τ
#
#   Stage 3 — binary (B=256, lr=1e-4) plus-shape around winner (α=0.3, τ=0.03):
#     9. α=0.3 τ=0.05          — τ-axis hole
#    10. α=0.3 τ=0.2           — τ-axis at winning α
#    11. α=0.3 τ=0.5           — τ-axis high side
#    12. α=0.5 τ=0.03          — α-axis at winning τ
#
# 12 jobs × 10-fold CV × 10h walltime each.
#
# (Supersedes slurm/run_jade_grid_holes.sh — covers all those cells plus extras.)
#
# Usage: bash slurm/run_jade_bulletproof.sh

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

echo "=== JADE bulletproof sweep — 12 jobs ==="
echo ""

echo "Stage 1 holes (B=128, lr=1e-4):"
submit binary  0.3 0.5  128 1e-4 "S1-main"
submit binary  0.9 0.1  128 1e-4 "S1-abl "

echo ""
echo "Stage 3 — 9-class plus-shape (B=256, lr=4e-4):"
submit 9-class 0.2 0.1  256 4e-4 "S3-9cl-α"
submit 9-class 0.2 0.2  256 4e-4 "S3-9cl-α"
submit 9-class 0.5 0.2  256 4e-4 "S3-9cl-α"
submit 9-class 0.3 0.05 256 4e-4 "S3-9cl-τ"
submit 9-class 0.3 0.5  256 4e-4 "S3-9cl-τ"

echo ""
echo "Stage 3 — 9-class LR cross-check (B=256, α=0.3 τ=0.2):"
submit 9-class 0.3 0.2  256 1e-4 "S3-9cl-LR"

echo ""
echo "Stage 3 — binary plus-shape (B=256, lr=1e-4):"
submit binary  0.3 0.05 256 1e-4 "S3-bin-τ"
submit binary  0.3 0.2  256 1e-4 "S3-bin-τ"
submit binary  0.3 0.5  256 1e-4 "S3-bin-τ"
submit binary  0.5 0.03 256 1e-4 "S3-bin-α"

echo ""
echo "All 12 jobs submitted. Monitor: squeue -u \$USER"
echo "Expected wall-clock: ~3 days at 2-concurrent limit."
