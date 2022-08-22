#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH --gres gpu:1

eval "$(conda shell.bash hook)"
bash
conda activate eg3d

python apply_factor.py --index 0 --output_dir output_v3_layer_loadd_all_FULL_wo_loss_direction_5 \
  --ckpt ./checkpoint_FFHQ_loadd_all_FULL_wo_loss_direction_5/552000.pt \
   --factor ./factor/factorv3_layer_loadd_all_FULL_wo_loss_direction_5.pt --ortho_id -2 --diag_size 5 # for all layers