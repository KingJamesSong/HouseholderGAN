#!/bin/bash
# Submit initialization ablation (svd_identity arm).
# random arm is already covered by 0214_ffhq128_autoenc_130M_multi_projector.
#
# Usage:
#   bash SUBMIT_0222_init_ablation.sh
#   INIT=random bash SUBMIT_0222_init_ablation.sh   # only if you want a re-run
set -e
cd /nfs/data_chaos/czhang/HouseholderGAN/diffae
INIT=${INIT:-svd_identity}

echo "=== init ablation: ${INIT} ==="
echo "Will save under checkpoints/0222_ffhq128_autoenc_130M_multi_projector_init_${INIT}/"
echo "Need ~8G free quota for 0/10k/25k/50k/100k (~2G each; step0 is lighter)."

J=$(sbatch --parsable --export=INIT=${INIT} ffhq128_init_ablation_train.sh)
echo "submitted train -> job $J"
J2=$(sbatch --parsable --dependency=afterany:$J --export=INIT=${INIT} ffhq128_init_ablation_train.sh)
echo "submitted resume backup -> job $J2"
squeue -u "$USER" | head
