#!/bin/bash
# JADE-LoRA — Stage A LR sweep, FOLD 1 ONLY, as a throttled SLURM job array.
#
# Same sweep as run_jade_lora_lr_fold1.sh, but submitted as ONE array job with
# at most 2 tasks running concurrently (--array=...%2). This respects the
# association GPU cap (gres/gpu=2 for education-eemcs-msc-cs): SLURM itself
# guarantees <=2 running tasks, so jobs queue cleanly instead of all starting
# and then being killed together with reason=AssocMaxGRESPerJob.
#
# Cells (5): 9-class {3e-4, 6e-3, 1e-2}, binary {6e-3, 1e-2}.
#   3e-4 = rerun of a previously-errored cell; 6e-3/1e-2 = upward LR extension.
#
# Fixed HPs (from winning JADE-FullFT loss HPs):
#   9-class : alpha=0.3 tau=0.2  B=256
#   binary  : alpha=0.2 tau=0.05 B=128
#   both    : LoRA rank=16, alpha=16 (scaling=1)
#
# Usage:  sbatch slurm/run_jade_lora_lr_fold1_array.sh
# Monitor: squeue -u $USER     (extra cells show PENDING reason=JobArrayTaskLimit)

#SBATCH --job-name="jadeLora-lrsweep-f1"
#SBATCH --partition=gpu-a100
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-cs
#SBATCH --output=slurm/logs/%A_%a.out
#SBATCH --error=slurm/logs/%A_%a.err
#SBATCH --array=0-4%2

set -e
cd ~/jade-mscthesis

JADE="src.approaches.jade.train_jade"
RANK=16

# ── Sweep table: one row per array index ─────────────────────────────────────
#                task     alpha tau   bs   ft_lr
CELLS=(
  "9-class   0.3   0.2   256  3e-4"   # 0
  "9-class   0.3   0.2   256  6e-3"   # 1
  "9-class   0.3   0.2   256  1e-2"   # 2
  "binary    0.2   0.05  128  6e-3"   # 3
  "binary    0.2   0.05  128  1e-2"   # 4
)

read -r TASK ALPHA TAU BS LR <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"

echo "=== JADE-LoRA LR sweep (array task $SLURM_ARRAY_TASK_ID) ==="
echo "Job:    ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node:   $(hostname)"
echo "Cell:   task=$TASK alpha=$ALPHA tau=$TAU bs=$BS ft_lr=$LR rank=$RANK"
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

echo "=== Running: python -m $JADE --dataset faced --task $TASK --fold 1 \\"
echo "    --lora-rank $RANK --alpha $ALPHA --temperature $TAU \\"
echo "    --batch-size $BS --ft-lr $LR --no-save-checkpoints ==="
echo ""

uv run python -m "$JADE" \
    --dataset faced --task "$TASK" --fold 1 \
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
