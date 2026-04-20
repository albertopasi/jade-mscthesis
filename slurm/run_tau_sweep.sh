#!/bin/bash
# Submit τ (temperature) sweep for JADE-FullFT on FACED.
#
# Fixed: fullft, pooling=no, repr=context, α ∈ {0.2, 0.3}, τ ∈ {0.05, 0.2, 0.5}
# τ=0.1 already run for both α values — not repeated here.
#
# 12 jobs: 3 τ × 2 α × 2 tasks
#
# Usage:
#   bash slurm/run_tau_sweep.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

MODULE="src.approaches.jade.train_jade"

echo "=== Submitting JADE τ sweep (FullFT, FACED, α∈{0.2,0.3}, τ∈{0.05,0.2,0.5}) ==="
echo ""

# ── binary, α=0.2 ─────────────────────────────────────────────────────────────

JOB=$(sbatch --job-name="jade-bin-a02-t005" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task binary --fullft --alpha 0.2 --temperature 0.05)
echo " 1/12 faced binary  fullft a=0.2 t=0.05: $JOB"

JOB=$(sbatch --job-name="jade-bin-a02-t02" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task binary --fullft --alpha 0.2 --temperature 0.2)
echo " 2/12 faced binary  fullft a=0.2 t=0.2:  $JOB"

JOB=$(sbatch --job-name="jade-bin-a02-t05" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task binary --fullft --alpha 0.2 --temperature 0.5)
echo " 3/12 faced binary  fullft a=0.2 t=0.5:  $JOB"

# ── binary, α=0.3 ─────────────────────────────────────────────────────────────

JOB=$(sbatch --job-name="jade-bin-a03-t005" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task binary --fullft --alpha 0.3 --temperature 0.05)
echo " 4/12 faced binary  fullft a=0.3 t=0.05: $JOB"

JOB=$(sbatch --job-name="jade-bin-a03-t02" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task binary --fullft --alpha 0.3 --temperature 0.2)
echo " 5/12 faced binary  fullft a=0.3 t=0.2:  $JOB"

JOB=$(sbatch --job-name="jade-bin-a03-t05" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task binary --fullft --alpha 0.3 --temperature 0.5)
echo " 6/12 faced binary  fullft a=0.3 t=0.5:  $JOB"

# ── 9-class, α=0.2 ────────────────────────────────────────────────────────────

JOB=$(sbatch --job-name="jade-9cl-a02-t005" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task 9-class --fullft --alpha 0.2 --temperature 0.05)
echo " 7/12 faced 9-class fullft a=0.2 t=0.05: $JOB"

JOB=$(sbatch --job-name="jade-9cl-a02-t02" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task 9-class --fullft --alpha 0.2 --temperature 0.2)
echo " 8/12 faced 9-class fullft a=0.2 t=0.2:  $JOB"

JOB=$(sbatch --job-name="jade-9cl-a02-t05" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task 9-class --fullft --alpha 0.2 --temperature 0.5)
echo " 9/12 faced 9-class fullft a=0.2 t=0.5:  $JOB"

# ── 9-class, α=0.3 ────────────────────────────────────────────────────────────

JOB=$(sbatch --job-name="jade-9cl-a03-t005" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task 9-class --fullft --alpha 0.3 --temperature 0.05)
echo "10/12 faced 9-class fullft a=0.3 t=0.05: $JOB"

JOB=$(sbatch --job-name="jade-9cl-a03-t02" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task 9-class --fullft --alpha 0.3 --temperature 0.2)
echo "11/12 faced 9-class fullft a=0.3 t=0.2:  $JOB"

JOB=$(sbatch --job-name="jade-9cl-a03-t05" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task 9-class --fullft --alpha 0.3 --temperature 0.5)
echo "12/12 faced 9-class fullft a=0.3 t=0.5:  $JOB"

echo ""
echo "All 12 τ sweep jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs will be in: slurm/logs/"
