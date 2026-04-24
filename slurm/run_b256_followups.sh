#!/bin/bash
# Follow-up B=256 runs after main sweep:
#   1. JADE binary @ lower LRs — does binary benefit from B=256 with the right LR?
#      (B=256 sweep tested 4e-4/8e-4, both worse than B=128 lr=1e-4. Binary likely
#       prefers smaller LR; test 1e-4 and 2e-4.)
#   2. FT-FullFT 9-class @ B=256, lr=4e-4 — isolates SupCon's contribution.
#      JADE 9-class jumped +4pp at B=256. Is it SupCon or the optimization recipe?
#   3. FT-FullFT binary @ B=256, lr ∈ {1e-4, 2e-4} — matching JADE-binary sweep
#      so any B=256 FT vs JADE comparison is at the same LR.
#
# All JADE and FT runs use --no-mixup for direct comparability (JADE disables mixup).
#
# 5 jobs × 10-fold CV. Walltime 10h each.
#
# Usage: bash slurm/run_b256_followups.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p slurm/logs

JADE="src.approaches.jade.train_jade"
FT="src.approaches.fine_tuning.train_ft"


echo ""
echo "FT-FullFT 9-class baseline @ B=256, lr=4e-4 (no mixup):"
JOB=$(sbatch --job-name="ft-9cl-b256" \
             --time=10:00:00 \
             slurm/run_experiment.sh $FT \
             --dataset faced --task 9-class --fullft --no-mixup \
             --batch-size 256 --ft-lr 4e-4)
echo "  ft-9cl-b256  $JOB"

echo ""
echo "FT-FullFT binary baseline @ B=256, LR sweep (no mixup):"
for LR in 1e-4 2e-4; do
    JOB=$(sbatch --job-name="ft-bin-lr${LR}" \
                 --time=10:00:00 \
                 slurm/run_experiment.sh $FT \
                 --dataset faced --task binary --fullft --no-mixup \
                 --batch-size 256 --ft-lr $LR)
    echo "  ft-bin-lr${LR}  $JOB"
done

echo ""
echo "All 5 jobs submitted. Monitor: squeue -u \$USER"
