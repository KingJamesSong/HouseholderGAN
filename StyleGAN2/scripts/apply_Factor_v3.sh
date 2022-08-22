#!/bin/bash
#SBATCH -p lambda
#SBATCH -A staff
#SBATCH --gres gpu:2

export PATH="/data/jzhang/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate styleganv2
python apply_factor.py --index 0 --output_dir output_v3_layer_loadd_all_partial \
  --ckpt ./checkpoint_FFHQ_loadd_all_partial/560000.pt \
   --factor ./factor/factorv3_layer_loadd_all_partial.pt --ortho_id -2