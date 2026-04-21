#!/bin/bash
# Self-contained smoke test for A100-80GB: does B=256 / B=512 fit?
# Single fold, 2 LP epochs + 3 FT epochs. Checks OOM + rough throughput.
#
# Usage:
#   sbatch --export=BS=256 slurm/run_batch_size_smoke.sh
#   sbatch --export=BS=512 slurm/run_batch_size_smoke.sh
# Or submit both in one go:
#   for bs in 256 512; do sbatch --export=BS=$bs slurm/run_batch_size_smoke.sh; done

#SBATCH --job-name="jade-smoke"
#SBATCH --partition=gpu-a100
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-cs
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err

BS="${BS:-256}"

echo "=== JADE batch-size smoke test ==="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Batch size: $BS"
echo "Started:    $(date)"
echo ""

# ── GPU info ─────────────────────────────────────────────────────────────────
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Background GPU logger (memory + util every 15s) ──────────────────────────
GPU_LOG="slurm/logs/${SLURM_JOB_ID}_gpu.csv"
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
    --format=csv -l 15 > "$GPU_LOG" &
GPU_LOG_PID=$!
trap 'kill $GPU_LOG_PID 2>/dev/null || true' EXIT

# ── Environment ──────────────────────────────────────────────────────────────
cd ~/jade-mscthesis

module load 2024r1
module load cuda/12.5

export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline
[ -f .env ] && export $(grep -v '^#' .env | xargs)

# ── Run ──────────────────────────────────────────────────────────────────────
echo "=== Running smoke test (bs=$BS) ==="
echo ""

uv run python -m src.approaches.jade.train_jade \
    --dataset faced --task 9-class --fullft \
    --alpha 0.3 --temperature 0.1 \
    --fold 1 --lp-epochs 2 --ft-epochs 3 \
    --batch-size "$BS"

EXIT_CODE=$?

echo ""
echo "=== Complete ==="
echo "Exit code:  $EXIT_CODE"
echo "GPU log:    $GPU_LOG"
echo "Peak mem:   $(awk -F', ' 'NR>1 {gsub(/ MiB/,\"\",$2); if ($2>m) m=$2} END {print m\" MiB\"}' $GPU_LOG)"
echo "Finished:   $(date)"

exit $EXIT_CODE
