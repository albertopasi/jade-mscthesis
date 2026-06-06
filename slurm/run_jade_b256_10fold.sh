#!/bin/bash
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

echo "=== JADE B=256 10-fold CV (FACED, 1 job) ==="
echo ""

echo "9-class (α=0.3, τ=0.1, lr=4e-4):"
submit 9-class 0.3 0.1 4e-4

echo ""
echo "1 job submitted. Monitor: squeue -u \$USER"
