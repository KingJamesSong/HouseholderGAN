#!/bin/bash
#SBATCH -p lambda
#SBATCH -A staff
#SBATCH --gres gpu:1

export PATH="/data/jzhang/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate styleganv2
python convert_weight.py --repo ~/stylegan2 stylegan2-ffhq-config-f.pkl --gen --disc