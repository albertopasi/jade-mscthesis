#!/bin/bash
# Submit first-wave JADE experiments (FACED only) to DelftBlue.
#
# Strategy — driven by FT results (docs/ft_results_analysis.md):
#   • pooling=no wins on 9-class & THU-EP (§7.3); keep as default.
#   • α (CE vs SupCon weight) is the most important new knob → sweep.
#   • Full FT + SupCon: 2 exploratory jobs only. Full FT memorises on CV
#     (train_loss→0) and has the worst gen ratio (§8.2); SupCon may or may not
#     rescue it. Keep it minimal until LoRA results are in.
#
# 12 jobs total. Run max 2 concurrent per account limit.
#
# Usage:
#   bash slurm/run_jade_experiments.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

MODULE="src.approaches.jade.train_jade"
GEN_SEEDS="--generalization --gen-seeds 123 456 789"

echo "=== Submitting JADE Jobs (FACED only) ==="
echo ""

# 1. Baseline: LoRA + SupCon α=0.5, context repr
# Mirrors the FT-LoRA no-mixup recipe + joint CE/SupCon. Reference point for
# everything else. Run both tasks × (CV, generalisation).

JOB=$(sbatch --job-name="jade-faced-bin" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task binary)
echo "1/12 faced binary  lora α=0.5 ctx  CV:           $JOB"

JOB=$(sbatch --job-name="jade-faced-9cl" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task 9-class)
echo "2/12 faced 9-class lora α=0.5 ctx  CV:           $JOB"

JOB=$(sbatch --job-name="jade-faced-bin-gen" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task binary $GEN_SEEDS)
echo "3/12 faced binary  lora α=0.5 ctx  GEN:          $JOB"

JOB=$(sbatch --job-name="jade-faced-9cl-gen" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task 9-class $GEN_SEEDS)
echo "4/12 faced 9-class lora α=0.5 ctx  GEN:          $JOB"

# 2. α sweep
# α=0.2 → SupCon dominates (stronger contrastive pull)
# α=0.8 → CE dominates (closer to FT baseline, SupCon as regulariser)
# Run on binary only for now

JOB=$(sbatch --job-name="jade-bin-a02" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task binary --alpha 0.2)
echo "5/12 faced binary  lora α=0.2 ctx  CV:           $JOB"

JOB=$(sbatch --job-name="jade-bin-a08" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task binary --alpha 0.8)
echo "6/12 faced binary  lora α=0.8 ctx  CV:           $JOB"

JOB=$(sbatch --job-name="jade-bin-a02-gen" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task binary --alpha 0.2 $GEN_SEEDS)
echo "7/12 faced binary  lora α=0.2 ctx  GEN:          $JOB"

JOB=$(sbatch --job-name="jade-bin-a08-gen" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task binary --alpha 0.8 $GEN_SEEDS)
echo "8/12 faced binary  lora α=0.8 ctx  GEN:          $JOB"

# Full FT + SupCon upper-bound probe (CV only) (α=0.5, context).

JOB=$(sbatch --job-name="jade-fullft-bin" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task binary --fullft)
echo "11/12 faced binary  fullft α=0.5 ctx CV:          $JOB"

JOB=$(sbatch --job-name="jade-fullft-9cl" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task 9-class --fullft)
echo "12/12 faced 9-class fullft α=0.5 ctx CV:          $JOB"

echo ""
echo "All 12 JADE jobs submitted. Monitor with: squeue --me"
echo "Logs will be in: slurm/logs/"
