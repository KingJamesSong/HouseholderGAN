#!/bin/bash
set -e
cd /nfs/data_chaos/czhang/HouseholderGAN/diffae
echo "Will resume from latest step=*.ckpt under checkpoints/0214_.../checkpoints/"
ls -lh checkpoints/0214_ffhq128_autoenc_130M_multi_projector/checkpoints/
J=$(sbatch --parsable ffhq128_steps_ablation_train.sh)
echo "submitted primary job: $J"
J2=$(sbatch --parsable --dependency=afterany:$J ffhq128_steps_ablation_train.sh)
echo "submitted resume job:  $J2"
squeue -u $USER | head
