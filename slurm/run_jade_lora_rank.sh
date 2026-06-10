#!/bin/bash
# JADE-LoRA — Stage B: rank sweep at FULL 10-fold CV (plain submission).
#
# LR was located in Stage A (fold-1 LR screen):
#   9-class : ft_lr = 3e-3   binary : ft_lr = 1e-3
#
# This sweeps LoRA rank {8, 16, 32} at full 10-fold CV with LR held fixed.
# --lora-alpha is left UNSET, so train_jade defaults lora_alpha = lora_rank and
# the LoRA scaling alpha/rank = 1 in every cell -> effective step size is
# constant across ranks, so the LR tuned at r16 transfers. Report as a table.
#
# Each run is full 10-fold CV. The pipeline already runs best-epoch inference
# after every fold and write_run_summary() pools it into the summary JSON with:
#   - confusion_matrix + per-class precision/recall/f1/support (window-level)
#   - subject_wise.{mean_acc, std_acc, min_acc, max_acc}  (std ACROSS SUBJECTS)
# So no --generalization and no checkpoint reload are needed.
#
# Fixed HPs (from winning JADE-FullFT loss HPs):
#   9-class : alpha=0.3 tau=0.2  B=256  ft_lr=3e-3
#   binary  : alpha=0.2 tau=0.05 B=128  ft_lr=1e-3
#
# 6 jobs (rank {8,16,32} x {9-class, binary}). Submitted one sbatch each; SLURM
# queues beyond the gres/gpu=2 cap and starts them as GPUs free up.
#
# Usage:  bash slurm/run_jade_lora_rank.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

JADE="src.approaches.jade.train_jade"

# ── Fixed HPs per task ───────────────────────────────────────────────────────
NINE_ALPHA=0.3;  NINE_TAU=0.2;   NINE_BS=256;  NINE_LR=3e-3
BIN_ALPHA=0.2;   BIN_TAU=0.05;   BIN_BS=128;   BIN_LR=1e-3
RANKS=(8 16 32)

submit() {
    local task=$1 alpha=$2 tau=$3 bs=$4 lr=$5 rank=$6
    local short="${task/9-class/9cl}"; short="${short/binary/bin}"
    local name="jadeLora-${short}-r${rank}-a${alpha}-t${tau}-b${bs}-lr${lr}"
    JOB=$(sbatch --job-name="$name" --time=10:00:00 \
                 slurm/run_experiment.sh $JADE \
                 --dataset faced --task $task \
                 --lora-rank $rank \
                 --alpha $alpha --temperature $tau \
                 --batch-size $bs --ft-lr $lr \
                 --no-save-checkpoints)
    echo "  $name  $JOB"
}

echo "=== JADE-LoRA Stage B: rank sweep, full 10-fold CV — 6 jobs ==="
echo ""
echo "9-class (alpha=$NINE_ALPHA, tau=$NINE_TAU, B=$NINE_BS, ft_lr=$NINE_LR):"
for r in "${RANKS[@]}"; do submit 9-class $NINE_ALPHA $NINE_TAU $NINE_BS $NINE_LR $r; done

echo ""
echo "binary (alpha=$BIN_ALPHA, tau=$BIN_TAU, B=$BIN_BS, ft_lr=$BIN_LR):"
for r in "${RANKS[@]}"; do submit binary $BIN_ALPHA $BIN_TAU $BIN_BS $BIN_LR $r; done

echo ""
echo "All 6 jobs submitted. Monitor: squeue -u \$USER"
echo "Results: outputs/jade_checkpoints/jade_lora_{9-class,binary}/summary_*.json"
echo "  -> confusion_matrix, per-class metrics, subject_wise.std_acc all included."
