#!/bin/bash
#SBATCH -p lambda
#SBATCH -A staff
#SBATCH --gres gpu:2

export PATH="/data/jzhang/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate styleganv2

python apply_factor.py --index 2 --output_dir output_v3_0 \
  --ckpt ./checkpoint_FFHQ_v3/551000.pt --factor ./factor/factorv3.pt --is_ortho