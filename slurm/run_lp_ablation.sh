#!/bin/bash
# LP ablation sweep on FACED.
#
# Axis 1 — window/stride (pool=no, mixup=on):
#   w5s2, w5s5, w8s4, w8s8, w10s5   (w10s10 baseline already done)
#
# Axis 2 — pooling (w10s10, mixup=on):
#   pool_last   (pool_no baseline already done; pool_mean skipped)
#
# Both binary and 9-class run sequentially within each job.
# All jobs use gpu-a100-small (10 GB) — LP has frozen encoder, <1 GB VRAM peak.
#
# Usage: bash slurm/run_lp_ablation.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

LP="src.approaches.linear_probing.train_lp"

echo "=== LP ablation — FACED (6 jobs) ==="
echo ""

# ── Axis 1: window sweep (pool=no, mixup=on) ─────────────────────────────────

echo "Window w5s2 (binary + 9-class):"
sbatch --job-name="lp-w5s2" \
       --partition=gpu-a100 \
       --time=2:00:00 \
       slurm/run_experiment.sh $LP \
       --dataset faced --task binary --window 5 --stride 2
sbatch --job-name="lp-w5s2-9cl" \
       --partition=gpu-a100 \
       --time=2:00:00 \
       slurm/run_experiment.sh $LP \
       --dataset faced --task 9-class --window 5 --stride 2
echo "  submitted w5s2 binary + 9-class"

echo "Window w5s5 (binary + 9-class):"
sbatch --job-name="lp-w5s5" \
       --partition=gpu-a100 \
       --time=2:00:00 \
       slurm/run_experiment.sh $LP \
       --dataset faced --task binary --window 5 --stride 5
sbatch --job-name="lp-w5s5-9cl" \
       --partition=gpu-a100 \
       --time=2:00:00 \
       slurm/run_experiment.sh $LP \
       --dataset faced --task 9-class --window 5 --stride 5
echo "  submitted w5s5 binary + 9-class"

echo "Window w8s4 (binary + 9-class):"
sbatch --job-name="lp-w8s4" \
       --partition=gpu-a100 \
       --time=2:00:00 \
       slurm/run_experiment.sh $LP \
       --dataset faced --task binary --window 8 --stride 4
sbatch --job-name="lp-w8s4-9cl" \
       --partition=gpu-a100 \
       --time=2:00:00 \
       slurm/run_experiment.sh $LP \
       --dataset faced --task 9-class --window 8 --stride 4
echo "  submitted w8s4 binary + 9-class"

echo "Window w8s8 (binary + 9-class):"
sbatch --job-name="lp-w8s8" \
       --partition=gpu-a100 \
       --time=2:00:00 \
       slurm/run_experiment.sh $LP \
       --dataset faced --task binary --window 8 --stride 8
sbatch --job-name="lp-w8s8-9cl" \
       --partition=gpu-a100 \
       --time=2:00:00 \
       slurm/run_experiment.sh $LP \
       --dataset faced --task 9-class --window 8 --stride 8
echo "  submitted w8s8 binary + 9-class"

echo "Window w10s5 (binary + 9-class):"
sbatch --job-name="lp-w10s5" \
       --partition=gpu-a100 \
       --time=2:00:00 \
       slurm/run_experiment.sh $LP \
       --dataset faced --task binary --window 10 --stride 5
sbatch --job-name="lp-w10s5-9cl" \
       --partition=gpu-a100 \
       --time=2:00:00 \
       slurm/run_experiment.sh $LP \
       --dataset faced --task 9-class --window 10 --stride 5
echo "  submitted w10s5 binary + 9-class"

# ── Axis 2: pooling (w10s10, mixup=on) ───────────────────────────────────────

echo ""
echo "Pooling pool_last w10s10 (binary + 9-class):"
sbatch --job-name="lp-last-bin" \
       --partition=gpu-a100 \
       --time=6:00:00 \
       slurm/run_experiment.sh $LP \
       --dataset faced --task binary --pooling last
sbatch --job-name="lp-last-9cl" \
       --partition=gpu-a100 \
       --time=6:00:00 \
       slurm/run_experiment.sh $LP \
       --dataset faced --task 9-class --pooling last
echo "  submitted pool_last binary + 9-class"

echo ""
echo "All 12 jobs submitted."
echo "Monitor: squeue -u \$USER"
