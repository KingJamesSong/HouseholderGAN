#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH --gres gpu:1

eval "$(conda shell.bash hook)"
bash
conda activate eg3d

python closed_form_factorization.py --out ./factor/factorv3_layer_loadd_all_FULL_wo_loss_direction_5.pt \
      ./checkpoint_FFHQ_loadd_all_FULL_wo_loss_direction_5/552000.pt --is_ortho --diag_size 5