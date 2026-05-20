#!/bin/bash
# Winning configs — re-run with all-fold checkpoint preservation, so
# per-class metrics (precision/recall/F1) and confusion matrices can
# be computed post-hoc via inference-only on the saved weights.
#
# Configs (from docs/full_report.md):
#   LP 9-class : w10s10, pool=no, --no-mixup            → 49.72 ± 2.93   (§7.1)
#   LP binary  : w10s10, pool=no, --no-mixup            → 71.50 ± 1.58   (§7.1)
#   FT 9-class : B=256, lr=4e-4, --no-mixup, --fullft   → 58.91 ± 2.95   (§8.2, JADE-matched)
#
# All checkpoints are kept on disk (per-fold subdirs); the auto-delete-
# non-best cleanup was removed from both LP and FT pipelines for this.
#
# Usage: bash slurm/run_winners_rerun.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

LP="src.approaches.linear_probing.train_lp"
FT="src.approaches.fine_tuning.train_ft"

submit_lp() {
    local task=$1
    local tag="${task/9-class/9cl}"; tag="${tag/binary/bin}"
    local name="lp-rerun-${tag}"
    JOB=$(sbatch --job-name="$name" \
                 --partition=gpu-a100 \
                 --time=1:30:00 \
                 slurm/run_experiment.sh $LP \
                 --dataset faced --task $task --no-mixup)
    echo "  $name  $JOB"
}

submit_ft() {
    local task=$1 batch=$2 lr=$3
    local tag="${task/9-class/9cl}"; tag="${tag/binary/bin}"
    local name="ft-rerun-${tag}-b${batch}-lr${lr}"
    JOB=$(sbatch --job-name="$name" \
                 --partition=gpu-a100 \
                 --time=10:00:00 \
                 slurm/run_experiment.sh $FT \
                 --dataset faced --fullft --no-mixup \
                 --task $task --batch-size $batch --ft-lr $lr)
    echo "  $name  $JOB"
}

echo "=== Winning configs re-run — 10-fold CV with checkpoint preservation ==="
echo ""

echo "LP:"
#submit_lp 9-class
#submit_lp binary

echo ""
echo "FT (9-class, JADE-matched recipe):"
submit_ft binary 128 1e-4

echo ""
echo "3 jobs submitted. Monitor: squeue -u \$USER"
