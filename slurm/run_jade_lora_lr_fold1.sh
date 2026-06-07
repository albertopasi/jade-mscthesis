#!/bin/bash
# JADE-LoRA — Stage A: learning-rate sweep, FOLD 1 ONLY (fast screen).
#
# Goal: the full-FT-optimal ft_lr is too low for LoRA (low-rank updates are
# scaled by alpha/r = 1, so the effective step is small). This screens a coarse
# half-decade LR grid at the current default rank (r=16, alpha=16) to locate the
# LoRA-appropriate LR region before committing to full 10-fold CV.
#
# Held fixed (per task), from the winning JADE-FullFT loss HPs:
#   9-class : supcon alpha=0.3, tau=0.2,  B=256
#   binary  : supcon alpha=0.2, tau=0.05, B=128
#   both    : LoRA rank=16, LoRA alpha=16 (=rank, scaling=1), target=attention
#
# Round-1 grid {1e-4, 3e-4, 1e-3, 3e-3} is already DONE except the 9-class 3e-4
# cell (transient error). This script now submits only the still-needed cells:
#   9-class : 3e-4 (rerun) + 6e-3, 1e-2  (upward extension — peak still rising at 3e-3)
#   binary  : 6e-3, 1e-2                 (upward extension — peak ~1e-3, expected to
#                                          confirm divergence; binary already collapses at 3e-3)
#
# Notes:
#   * fold 1 only -> writes summary_..._lr{lr}_fold1.json into
#     outputs/jade_checkpoints/ (single-fold path; std=0.0, n_folds_run=1).
#     Use these only to discard bad/diverging LRs and pick the top 1-2 region;
#     single-fold acc has ~1pp noise, so confirm the final LR at full 10-fold CV.
#   * --no-save-checkpoints -> no LoRA adapter / head / proj-head written to disk.
#   * LoRA is the default (no --fullft).
#
# 5 jobs (9-class x3, binary x2), 1h walltime each.
#
# Usage: bash slurm/run_jade_lora_lr_fold1.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

JADE="src.approaches.jade.train_jade"

# ── Fixed HPs per task (edit here) ───────────────────────────────────────────
NINE_ALPHA=0.3;  NINE_TAU=0.2;   NINE_BS=256    # 9-class loss HPs + batch size
BIN_ALPHA=0.2;   BIN_TAU=0.05;   BIN_BS=128     # binary  loss HPs + batch size
RANK=16                            # alpha defaults to rank -> scaling = 1

# LRs still to submit per task (round-1 {1e-4,3e-4,1e-3,3e-3} already done,
# except the 9-class 3e-4 cell which errored and is re-run here).
NINE_LRS=(3e-4 6e-3 1e-2)   # 3e-4 = rerun; 6e-3,1e-2 = upward extension
BIN_LRS=(6e-3 1e-2)         # upward extension only

submit() {
    local task=$1 alpha=$2 tau=$3 bs=$4 lr=$5
    local short="${task/9-class/9cl}"; short="${short/binary/bin}"
    local name="jadeLora-${short}-r${RANK}-a${alpha}-t${tau}-b${bs}-lr${lr}-f1"
    JOB=$(sbatch --job-name="$name" --time=01:00:00 \
                 slurm/run_experiment.sh $JADE \
                 --dataset faced --task $task --fold 1 \
                 --lora-rank $RANK \
                 --alpha $alpha --temperature $tau \
                 --batch-size $bs --ft-lr $lr \
                 --no-save-checkpoints)
    echo "  $name  $JOB"
}

echo "=== JADE-LoRA Stage A: LR sweep extension (fold 1, r=$RANK) — 5 jobs ==="
echo ""
echo "9-class (alpha=$NINE_ALPHA, tau=$NINE_TAU, B=$NINE_BS):"
for lr in "${NINE_LRS[@]}"; do submit 9-class $NINE_ALPHA $NINE_TAU $NINE_BS $lr; done

echo ""
echo "binary (alpha=$BIN_ALPHA, tau=$BIN_TAU, B=$BIN_BS):"
for lr in "${BIN_LRS[@]}"; do submit binary $BIN_ALPHA $BIN_TAU $BIN_BS $lr; done

echo ""
echo "All 5 jobs submitted. Monitor: squeue -u \$USER"
echo "Results: outputs/jade_checkpoints/summary_*_fold1.json  +  W&B (offline)  +  slurm/logs/<jobid>.out"
