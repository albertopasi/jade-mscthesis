#!/bin/bash
# Submit FT experiments: pooling=last, no mixup.
# 4 jobs: FACED & THU-EP × binary & 9-class.
#
# Usage:
#   bash slurm/run_ft_experiments_last_nomixup.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

MODULE="src.approaches.fine_tuning.train_ft"

echo "=== Submitting FT last-pool no-mixup Jobs ==="
echo ""

# ── FACED ─────────────────────────────────────────────────────────────────

# 1. FACED — last pooling, binary, no mixup
JOB1=$(sbatch --job-name="ft-faced-last-bin-nm" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task binary --pooling last --no-mixup)
echo "1/4  faced last binary nomixup:   $JOB1"

# 2. FACED — last pooling, 9-class, no mixup
JOB2=$(sbatch --job-name="ft-faced-last-9cl-nm" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task 9-class --pooling last --no-mixup)
echo "2/4  faced last 9-class nomixup:  $JOB2"

# ── THU-EP ────────────────────────────────────────────────────────────────

# 3. THU-EP — last pooling, binary, no mixup
JOB3=$(sbatch --job-name="ft-thuep-last-bin-nm" \
    slurm/run_experiment.sh $MODULE \
    --dataset thu-ep --task binary --pooling last --no-mixup)
echo "3/4  thu-ep last binary nomixup:  $JOB3"

# 4. THU-EP — last pooling, 9-class, no mixup
JOB4=$(sbatch --job-name="ft-thuep-last-9cl-nm" \
    slurm/run_experiment.sh $MODULE \
    --dataset thu-ep --task 9-class --pooling last --no-mixup)
echo "4/4  thu-ep last 9-class nomixup: $JOB4"

echo ""
echo "All 4 jobs submitted. Monitor with: squeue --me"
echo "Logs will be in: slurm/logs/"
