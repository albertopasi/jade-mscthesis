#!/bin/bash
# Submit all FT experiments (regular + generalization) to DelftBlue.
#
# 8 jobs total: FACED & THU-EP × binary & 9-class × regular & generalization.
# Generalization runs use 3 seeds (123 456 789), all 10 folds each.
#
# Usage:
#   bash slurm/run_ft_experiments.sh
#
# Note: can run max 2 GPU jobs at once (account limit),
# so later jobs will queue until earlier ones finish.

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

MODULE="src.approaches.fine_tuning.train_ft"
GEN_SEEDS="--generalization --gen-seeds 123 456 789"

echo "=== Submitting FT Jobs ==="
echo ""

# ── FACED ─────────────────────────────────────────────────────────────────

# 1. FACED — no pooling, binary
JOB1=$(sbatch --job-name="ft-faced-bin" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task binary)
echo "1/8  faced no-pool binary:   $JOB1"

# 2. FACED — no pooling, 9-class
JOB2=$(sbatch --job-name="ft-faced-9cl" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task 9-class)
echo "2/8  faced no-pool 9-class:  $JOB2"

# 3. FACED — no pooling, binary, gen
JOB3=$(sbatch --job-name="ft-faced-bin-gen" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task binary \
    $GEN_SEEDS)
echo "3/8  faced binary gen: $JOB3"

# 4. FACED — last pooling, 9-class, gen
JOB4=$(sbatch --job-name="ft-faced-9cl-gen" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task 9-class --pooling last \
    $GEN_SEEDS)
echo "4/8  faced 9-class gen: $JOB4"

# ── THU-EP ────────────────────────────────────────────────────────────────

# 5. THU-EP — no pooling, binary
JOB5=$(sbatch --job-name="ft-thuep-bin" \
    slurm/run_experiment.sh $MODULE \
    --dataset thu-ep --task binary)
echo "5/8  thu-ep no-pool binary:   $JOB5"

# 6. THU-EP — no pooling, 9-class
JOB6=$(sbatch --job-name="ft-thuep-no-9cl" \
    slurm/run_experiment.sh $MODULE \
    --dataset thu-ep --task 9-class)
echo "6/8  thu-ep no-pool 9-class:  $JOB6"

# 7. THU-EP — no pooling, binary, gen
JOB7=$(sbatch --job-name="ft-thuep-bin-gen" \
    slurm/run_experiment.sh $MODULE \
    --dataset thu-ep --task binary \
    $GEN_SEEDS)
echo "7/8  thu-ep binary gen:      $JOB7"

# 8. THU-EP — no pooling, 9-class, gen
JOB8=$(sbatch --job-name="ft-thuep-9cl-gen" \
    slurm/run_experiment.sh $MODULE \
    --dataset thu-ep --task 9-class \
    $GEN_SEEDS)
echo "8/8  thu-ep 9-class gen:     $JOB8"

echo ""
echo "All 8 jobs submitted. Monitor with: squeue --me"
echo "Logs will be in: slurm/logs/"
