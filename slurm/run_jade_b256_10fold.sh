#!/bin/bash
# JADE-FullFT 10-fold CV @ B=256 on FACED.
#
# For each task we fix α at its optimum (from B=128 sweep) and test τ
# sensitivity around it at two LRs.
#
# Binary:  α=0.2, τ ∈ {0.05, 0.1}    — B=128 optimum was (0.2, 0.05)
# 9-class: α=0.3, τ ∈ {0.1,  0.2}    — B=128 optimum was (0.3, 0.1)
# LRs:     4e-4 (safe), 8e-4 (aggressive)
#
# Grid: 2 tasks × 2 τ × 2 LR = 8 jobs × 10 folds.
# Walltime: 10h/job (~45min × 10 folds + margin).
#
# Usage: bash slurm/run_jade_b256_10fold.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

MODULE="src.approaches.jade.train_jade"
BASE="--dataset faced --fullft --batch-size 256"

submit() {
    local task=$1 alpha=$2 tau=$3 lr=$4
    local tag="${task/9-class/9cl}"; tag="${tag/binary/bin}"
    local name="jade-${tag}-a${alpha}-t${tau}-lr${lr}"
    JOB=$(sbatch --job-name="$name" \
                 --time=10:00:00 \
                 slurm/run_experiment.sh $MODULE $BASE \
                 --task $task --alpha $alpha --temperature $tau --ft-lr $lr)
    echo "  $name  $JOB"
}

echo "=== JADE B=256 10-fold CV (FACED, 8 jobs) ==="
echo ""

echo "Binary (α=0.2):"
for TAU in 0.05 0.1; do
    for LR in 4e-4 8e-4; do
        submit binary 0.2 $TAU $LR
    done
done

echo ""
echo "9-class (α=0.3):"
for TAU in 0.1 0.2; do
    for LR in 4e-4 8e-4; do
        submit 9-class 0.3 $TAU $LR
    done
done

echo ""
echo "All 8 jobs submitted. Monitor: squeue -u \$USER"
