#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH --gres gpu:1

eval "$(conda shell.bash hook)"
bash
conda activate eg3d

python test_VP_pairs.py --ortho_id -1 --size 256 \
        --ckpt ./checkpoint_FFHQ_loadd_all_FULL_wo_loss_b16/552000.pt  \
        --factor ./factor/factor_SaFA.pt --output_dir output_vp_pairs_wo_loss_direction_5_b8