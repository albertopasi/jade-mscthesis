#!/bin/bash
# JADE-FullFT 10-fold CV reruns of the two thesis winning configurations,
# with the updated pipeline that collects per-window (y_true, y_pred) on the
# val set at every fold's best checkpoint. The summary JSON gains a pooled
# `classification_report` (confusion matrix + per-class precision/recall/F1)
# plus `predictions_by_fold` for post-hoc analysis.
#
# Configs (from docs/full_report.md §9.4):
#   9-class : α=0.3, τ=0.2,  B=256, ft_lr=4e-4
#   Binary  : α=0.2, τ=0.05, B=128, ft_lr=1e-4 (REVE-default recipe)
#
# 2 jobs × 10 folds. Walltime: 10h/job.
#
# Usage: bash slurm/run_jade_winners_confmat.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

MODULE="src.approaches.jade.train_jade"
BASE="--dataset faced --fullft"

submit() {
    local task=$1 alpha=$2 tau=$3 batch=$4 lr=$5
    local tag="${task/9-class/9cl}"; tag="${tag/binary/bin}"
    local name="jade-cm-${tag}-a${alpha}-t${tau}-b${batch}-lr${lr}"
    JOB=$(sbatch --job-name="$name" \
                 --time=10:00:00 \
                 slurm/run_experiment.sh $MODULE $BASE \
                 --task $task --alpha $alpha --temperature $tau \
                 --batch-size $batch --ft-lr $lr)
    echo "  $name  $JOB"
}

echo "=== JADE winning configs — 10-fold CV with confusion-matrix collection ==="
echo ""

# 9-class winner
submit 9-class 0.3 0.2  256 4e-4

# Binary winner
submit binary  0.2 0.05 128 1e-4

echo ""
echo "2 jobs submitted. Monitor: squeue -u \$USER"
