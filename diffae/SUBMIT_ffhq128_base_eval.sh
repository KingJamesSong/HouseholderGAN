#!/bin/bash
# Submit FID + PPL/PIPL for original DiffAE (ffhq128_autoenc_130M).
set -e
cd /nfs/data_chaos/czhang/HouseholderGAN/diffae

ls -lh checkpoints/ffhq128_autoenc_130M/last.ckpt

J1=$(sbatch --parsable ffhq128_base_fid_eval.sh)
echo "submitted FID  -> job $J1"

J2=$(sbatch --parsable ffhq128_base_ppl_pipl.sh)
echo "submitted PPL/PIPL -> job $J2"

squeue -u "$USER" | head
