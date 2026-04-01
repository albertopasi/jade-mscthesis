#!/bin/bash
# Submit missing FT experiments (excluding revesplit, all no-mixup).
#
# Missing configurations:
#   1.  faced   9-class  pool_last  r16     nomixup  (10-fold)
#   2.  faced   binary   gen        r16     nomixup  (3 seeds)
#   3.  faced   9-class  gen        r16     nomixup  (3 seeds)
#   4.  thu-ep  binary   fullft             nomixup  (10-fold)
#   5.  thu-ep  9-class  fullft             nomixup  (10-fold)
#   6.  thu-ep  binary   gen        r16     nomixup  (3 seeds)
#   7.  thu-ep  9-class  gen        r16     nomixup  (3 seeds)
#   8.  faced   binary   gen        fullft  nomixup  (3 seeds)
#   9.  faced   9-class  gen        fullft  nomixup  (3 seeds)
#   10. thu-ep  binary   gen        fullft  nomixup  (3 seeds)
#   11. thu-ep  9-class  gen        fullft  nomixup  (3 seeds)
#
# Usage:
#   bash slurm/run_missing_experiments.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

MODULE="src.approaches.fine_tuning.train_ft"

echo "=== Submitting Missing FT Jobs (all no-mixup) ==="
echo ""

# 1. FACED — 9-class, pool_last, LoRA r16, no-mixup, 10-fold
JOB1=$(sbatch --job-name="ft-faced-9cl-last-nm" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task 9-class --pooling last --no-mixup)
echo " 1/11  faced 9-class pool_last r16 nomixup 10-fold:     $JOB1"

# 2. FACED — binary, pool_no, LoRA r16, no-mixup, generalization (3 seeds)
JOB2=$(sbatch --job-name="ft-faced-bin-gen-nm" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task binary --generalization --gen-seeds 123 456 789 --no-mixup)
echo " 2/11  faced binary  pool_no  r16 nomixup gen:          $JOB2"

# 3. FACED — 9-class, pool_no, LoRA r16, no-mixup, generalization (3 seeds)
JOB3=$(sbatch --job-name="ft-faced-9cl-gen-nm" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task 9-class --generalization --gen-seeds 123 456 789 --no-mixup)
echo " 3/11  faced 9-class pool_no  r16 nomixup gen:          $JOB3"

# 4. THU-EP — binary, full FT, no-mixup, 10-fold
JOB4=$(sbatch --job-name="ft-thuep-bin-fft-nm" \
    slurm/run_experiment.sh $MODULE \
    --dataset thu-ep --task binary --fullft --no-mixup)
echo " 4/11  thu-ep binary  fullft nomixup 10-fold:           $JOB4"

# 5. THU-EP — 9-class, full FT, no-mixup, 10-fold
JOB5=$(sbatch --job-name="ft-thuep-9cl-fft-nm" \
    slurm/run_experiment.sh $MODULE \
    --dataset thu-ep --task 9-class --fullft --no-mixup)
echo " 5/11  thu-ep 9-class fullft nomixup 10-fold:           $JOB5"

# 6. THU-EP — binary, pool_no, LoRA r16, no-mixup, generalization (3 seeds)
JOB6=$(sbatch --job-name="ft-thuep-bin-gen-nm" \
    slurm/run_experiment.sh $MODULE \
    --dataset thu-ep --task binary --generalization --gen-seeds 123 456 789 --no-mixup)
echo " 6/11  thu-ep binary  pool_no  r16 nomixup gen:         $JOB6"

# 7. THU-EP — 9-class, pool_no, LoRA r16, no-mixup, generalization (3 seeds)
JOB7=$(sbatch --job-name="ft-thuep-9cl-gen-nm" \
    slurm/run_experiment.sh $MODULE \
    --dataset thu-ep --task 9-class --generalization --gen-seeds 123 456 789 --no-mixup)
echo " 7/11  thu-ep 9-class pool_no  r16    nomixup gen:         $JOB7"

# 8. FACED — binary, full FT, no-mixup, generalization (3 seeds)
JOB8=$(sbatch --job-name="ft-faced-bin-fft-gen-nm" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task binary --fullft --generalization --gen-seeds 123 456 789 --no-mixup)
echo " 8/11  faced binary  fullft nomixup gen:                $JOB8"

# 9. FACED — 9-class, full FT, no-mixup, generalization (3 seeds)
JOB9=$(sbatch --job-name="ft-faced-9cl-fft-gen-nm" \
    slurm/run_experiment.sh $MODULE \
    --dataset faced --task 9-class --fullft --generalization --gen-seeds 123 456 789 --no-mixup)
echo " 9/11  faced 9-class fullft nomixup gen:                $JOB9"

# 10. THU-EP — binary, full FT, no-mixup, generalization (3 seeds)
JOB10=$(sbatch --job-name="ft-thuep-bin-fft-gen-nm" \
    slurm/run_experiment.sh $MODULE \
    --dataset thu-ep --task binary --fullft --generalization --gen-seeds 123 456 789 --no-mixup)
echo "10/11  thu-ep binary  fullft nomixup gen:               $JOB10"

# 11. THU-EP — 9-class, full FT, no-mixup, generalization (3 seeds)
JOB11=$(sbatch --job-name="ft-thuep-9cl-fft-gen-nm" \
    slurm/run_experiment.sh $MODULE \
    --dataset thu-ep --task 9-class --fullft --generalization --gen-seeds 123 456 789 --no-mixup)
echo "11/11  thu-ep 9-class fullft nomixup gen:               $JOB11"

echo ""
echo "All 11 jobs submitted. Monitor with: squeue --me"
echo "Logs will be in: slurm/logs/"
