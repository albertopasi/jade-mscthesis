#!/bin/bash
# JADE-LoRA — Stage B: rank sweep at FULL 10-fold CV, as a throttled job array.
#
# LR was located in Stage A (run_jade_lora_lr_fold1_array.sh, fold-1 screen):
#   9-class : ft_lr = 3e-3  (peak; collapses at >=6e-3)
#   binary  : ft_lr = 1e-3  (peak; collapses at >=3e-3)
#
# This sweeps LoRA rank {8, 16, 32} at full 10-fold CV with the LR held fixed.
# alpha defaults to rank (train_jade.py: lora_alpha = lora_rank if unset), so
# the LoRA scaling alpha/rank = 1 in EVERY cell -> effective step size is constant
# across ranks, so the LR tuned at r16 transfers. Report the 6 cells as a table.
#
# Fixed HPs (from winning JADE-FullFT loss HPs):
#   9-class : alpha=0.3 tau=0.2  B=256  ft_lr=3e-3
#   binary  : alpha=0.2 tau=0.05 B=128  ft_lr=1e-3
#
# 6 cells = rank {8,16,32} x {9-class, binary}. Throttled to 2 concurrent (%2)
# to respect the association GPU cap (gres/gpu=2 for education-eemcs-msc-cs):
# SLURM holds extra tasks PENDING (reason=JobArrayTaskLimit) instead of starting
# them all and getting killed together with reason=AssocMaxGRESPerJob.
#
# Full 10-fold CV is long; --time is per task (not for the whole array).
#
# Usage:  sbatch slurm/run_jade_lora_rank_array.sh
# Monitor: squeue -u $USER     (extra cells show PENDING reason=JobArrayTaskLimit)

#SBATCH --job-name="jadeLora-ranksweep"
#SBATCH --partition=gpu-a100
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-cs
#SBATCH --output=slurm/logs/%A_%a.out
#SBATCH --error=slurm/logs/%A_%a.err
#SBATCH --array=0-5%2

set -e
cd ~/jade-mscthesis

JADE="src.approaches.jade.train_jade"

# ── Sweep table: one row per array index ─────────────────────────────────────
# alpha (LoRA) is left unset -> defaults to rank, so scaling alpha/rank = 1.
#                task     s_alpha s_tau  bs   ft_lr  rank
CELLS=(
  "9-class   0.3   0.2   256  3e-3   8"    # 0
  "9-class   0.3   0.2   256  3e-3   16"   # 1
  "9-class   0.3   0.2   256  3e-3   32"   # 2
  "binary    0.2   0.05  128  1e-3   8"    # 3
  "binary    0.2   0.05  128  1e-3   16"   # 4
  "binary    0.2   0.05  128  1e-3   32"   # 5
)

read -r TASK ALPHA TAU BS LR RANK <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"

echo "=== JADE-LoRA rank sweep (array task $SLURM_ARRAY_TASK_ID) ==="
echo "Job:    ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node:   $(hostname)"
echo "Cell:   task=$TASK supcon_alpha=$ALPHA tau=$TAU bs=$BS ft_lr=$LR rank=$RANK (lora_alpha=rank)"
echo "Started: $(date)"
echo ""

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Environment (matches run_experiment.sh) ──────────────────────────────────
module load 2024r1
module load cuda/12.5
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline
[ -f .env ] && export $(grep -v '^#' .env | xargs)

echo "=== Running: python -m $JADE --dataset faced --task $TASK \\"
echo "    --lora-rank $RANK --alpha $ALPHA --temperature $TAU \\"
echo "    --batch-size $BS --ft-lr $LR --no-save-checkpoints ==="
echo ""

uv run python -m "$JADE" \
    --dataset faced --task "$TASK" \
    --lora-rank "$RANK" \
    --alpha "$ALPHA" --temperature "$TAU" \
    --batch-size "$BS" --ft-lr "$LR" \
    --no-save-checkpoints

EXIT_CODE=$?
echo ""
echo "=== Complete ==="
echo "Exit code: $EXIT_CODE"
echo "Finished:  $(date)"
exit $EXIT_CODE
