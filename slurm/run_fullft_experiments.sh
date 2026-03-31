#!/bin/bash
# Submit full fine-tuning experiments (no LoRA) for FACED 9-class.
#
# 2 jobs:
#   1. Standard 10-fold cross-validation
#   2. Official REVE static split (train 0-79, val 80-99, test 100-122)
#
# Usage:
#   bash slurm/run_fullft_experiments.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

MODULE="src.approaches.fine_tuning.train_ft"

echo "=== Submitting Full FT Jobs (FACED 9-class) ==="
echo ""

# 1. FACED — 9-class, full FT, 10-fold cross-validation
JOB1=$(sbatch --job-name="fullft-faced-9cl" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task 9-class --fullft)
echo "1/2  faced 9-class fullft 10-fold:    $JOB1"

# 2. FACED — 9-class, full FT, official REVE static split
JOB2=$(sbatch --job-name="fullft-faced-9cl-rs" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task 9-class --fullft --revesplit)
echo "2/2  faced 9-class fullft revesplit:  $JOB2"

echo ""
echo "All 2 jobs submitted. Monitor with: squeue --me"
echo "Logs will be in: slurm/logs/"
