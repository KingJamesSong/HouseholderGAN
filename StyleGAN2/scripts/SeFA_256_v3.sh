#!/bin/bash
#SBATCH -p lambda
#SBATCH -A staff
#SBATCH --gres gpu:2

export PATH="/data/jzhang/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate styleganv2
python closed_form_factorization.py --out ./factor/factorv3_layer_loadd_all_FULL.pt \
      ./checkpoint_FFHQ_loadd_all_FULL/555000.pt --is_ortho