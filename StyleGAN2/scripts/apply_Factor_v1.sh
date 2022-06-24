#!/bin/bash
#SBATCH -p lambda
#SBATCH -A staff
#SBATCH --gres gpu:2

export PATH="/data/jzhang/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate styleganv2

python apply_factor.py --index 0 --output_dir output_v1_0 --ckpt ./checkpoint_FFHQ/551000.pt --factor ./factor/factorv2.pt