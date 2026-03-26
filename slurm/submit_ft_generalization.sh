#!/bin/bash
# Submit all 4 FT generalization experiments to DelftBlue.
#
# Runs: w10s10, 3 seeds (123 456 789), all 10 folds each.
#   1. pool binary
#   2. pool 9-class
#   3. nopool_flat binary
#   4. nopool_flat 9-class
#
# Usage:
#   bash slurm/submit_ft_generalization.sh
#
# Note: you can run max 2 GPU jobs at once (account limit),
# so jobs 3 and 4 will queue until 1 and 2 finish.

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

MODULE="src.approaches.lora_finetuning.train_lora"
GEN_SEEDS="--generalization --gen-seeds 123 456 789"

echo "=== Submitting FT Generalization Jobs ==="
echo ""

# 1. Pool binary
JOB1=$(sbatch --job-name="ft-gen-pool-bin" \
    slurm/run_experiment.sh $MODULE \
    --task binary --window 10 --stride 10 \
    $GEN_SEEDS)
echo "1/4  pool binary:        $JOB1"

# 2. Pool 9-class
JOB2=$(sbatch --job-name="ft-gen-pool-9cl" \
    slurm/run_experiment.sh $MODULE \
    --task 9-class --window 10 --stride 10 \
    $GEN_SEEDS)
echo "2/4  pool 9-class:       $JOB2"

# 3. Nopool flat binary
JOB3=$(sbatch --job-name="ft-gen-flat-bin" \
    slurm/run_experiment.sh $MODULE \
    --task binary --window 10 --stride 10 \
    --no-pooling --no-pool-mode flat \
    $GEN_SEEDS)
echo "3/4  nopool_flat binary: $JOB3"

# 4. Nopool flat 9-class
JOB4=$(sbatch --job-name="ft-gen-flat-9cl" \
    slurm/run_experiment.sh $MODULE \
    --task 9-class --window 10 --stride 10 \
    --no-pooling --no-pool-mode flat \
    $GEN_SEEDS)
echo "4/4  nopool_flat 9-class: $JOB4"

echo ""
echo "All 4 jobs submitted. Monitor with: squeue --me"
echo "Logs will be in: slurm/logs/"
