#!/bin/bash
# Generic SLURM job script for running experiments on DelftBlue.
#
# Usage:
#   sbatch slurm/run_experiment.sh <module> [args...]
#
# Examples:
#   # FT generalization — pool binary
#   sbatch slurm/run_experiment.sh src.approaches.fine_tuning.train_ft \
#       --task binary --generalization --gen-seeds 123 456 789
#
#   # FT single fold, THU-EP, 9-class
#   sbatch slurm/run_experiment.sh src.approaches.fine_tuning.train_ft \
#       --dataset thu-ep --task 9-class --fold 1
#
#   # LP all folds, FACED, binary
#   sbatch slurm/run_experiment.sh src.approaches.linear_probing.train_lp \
#       --dataset faced --task binary

#SBATCH --job-name="eeg-exp"
#SBATCH --partition=gpu-a100
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-cs
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err

# ── Parse arguments ──────────────────────────────────────────────────────────
MODULE="$1"
shift
ARGS="$@"

if [ -z "$MODULE" ]; then
    echo "ERROR: No module specified."
    echo "Usage: sbatch slurm/run_experiment.sh <module> [args...]"
    exit 1
fi

echo "=== Experiment on DelftBlue ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Module:    $MODULE"
echo "Args:      $ARGS"
echo "Started:   $(date)"
echo ""

# ── GPU info ─────────────────────────────────────────────────────────────────
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Environment ──────────────────────────────────────────────────────────────
cd ~/jade-mscthesis

module load 2024r1
module load cuda/12.5

export PATH="$HOME/.local/bin:$PATH"

# W&B: GPU nodes have no internet — log offline, sync later from login node
export WANDB_MODE=offline
# Load API key from .env (gitignored) if it exists
[ -f .env ] && export $(grep -v '^#' .env | xargs)

echo "=== Environment ==="
echo "Python:  $(uv run python --version)"
echo "PyTorch: $(uv run python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(uv run python -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)')"
echo ""

# ── Run ──────────────────────────────────────────────────────────────────────
echo "=== Running: python -m $MODULE $ARGS ==="
echo ""

uv run python -m $MODULE $ARGS

EXIT_CODE=$?

echo ""
echo "=== Complete ==="
echo "Exit code: $EXIT_CODE"
echo "Finished:  $(date)"

exit $EXIT_CODE
