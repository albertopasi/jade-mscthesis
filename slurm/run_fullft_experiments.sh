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

echo "=== Submitting Full FT Jobs (FACED 9-class and binary) ==="
echo ""

# 1. FACED — 9-class, full FT, 10-fold cross-validation
JOB1=$(sbatch --job-name="fullft-faced-9cl" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task 9-class --fullft)
echo "1/4  faced 9-class fullft 10-fold:    $JOB1"

# 2. FACED — 9-class, full FT, official REVE static split
JOB2=$(sbatch --job-name="fullft-faced-9cl-rs" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task 9-class --fullft --revesplit)
echo "2/4  faced 9-class fullft revesplit:  $JOB2"

# 3. FACED — binary, full FT, 10-fold cross-validation
JOB3=$(sbatch --job-name="fullft-faced-bin" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task binary --fullft)
echo "3/4  faced 9-class fullft 10-fold:    $JOB3"

# 4. FACED — binary, full FT, official REVE static split
JOB4=$(sbatch --job-name="fullft-faced-bin-rs" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task binary --fullft --revesplit)
echo "4/4  faced 9-class fullft revesplit:  $JOB4"

echo ""
echo "All 4 jobs submitted. Monitor with: squeue --me"
echo "Logs will be in: slurm/logs/"
