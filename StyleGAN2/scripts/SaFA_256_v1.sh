#!/bin/bash
#SBATCH -p lambda
#SBATCH -A staff
#SBATCH --gres gpu:2

export PATH="/data/jzhang/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate styleganv2

python closed_form_factorization.py --out ./factor/factorv2.pt ./checkpoint_FFHQv3/551000.pt