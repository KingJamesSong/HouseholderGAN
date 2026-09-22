#!/bin/bash
set -e
cd /nfs/data_chaos/czhang/HouseholderGAN/diffae

echo "=== checkpoints ==="
ls -lh checkpoints/0214_ffhq128_autoenc_130M_multi_projector/checkpoints/

# 1) FID eval for step=50000
JE=$(sbatch --parsable --export=STEP=50000 ffhq128_steps_ablation_eval.sh)
echo "submitted FID eval step=50000 -> job $JE"

# 2) resume train to 100k (from step=50000.ckpt)
JT=$(sbatch --parsable ffhq128_steps_ablation_train.sh)
echo "submitted train resume -> job $JT"
# safety net if walltime cuts before 100k
JT2=$(sbatch --parsable --dependency=afterany:$JT ffhq128_steps_ablation_train.sh)
echo "submitted train resume backup -> job $JT2"

squeue -u "$USER" | head
