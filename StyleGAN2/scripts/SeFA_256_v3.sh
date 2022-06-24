#!/bin/bash
#SBATCH -p lambda
#SBATCH -A staff
#SBATCH --gres gpu:2

export PATH="/data/jzhang/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate styleganv2

python closed_form_factorization.py --out ./factor/factorv3.pt ./checkpoint_FFHQ_v3/551000.pt --is_ortho