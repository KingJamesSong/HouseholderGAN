#!/bin/bash
#SBATCH -p lambda
#SBATCH -A staff
#SBATCH --gres gpu:2

export PATH="/data/jzhang/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate styleganv2

python apply_factor.py --index 0 --output_dir output_SeFA4 --ckpt ./checkpoint_FFHQ/550000.pt --factor ./factor/factor_SaFA4.pt --ortho_id -1