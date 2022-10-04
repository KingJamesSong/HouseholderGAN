#!/bin/bash
#SBATCH -p lambda
#SBATCH -A staff
#SBATCH --gres gpu:1

export PATH="/data/jzhang/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate eg3d

python calc_metrics.py --metrics=fid50k_full --data=/nfs/data_chaos/datasets/FFHQ1024/ffhq-1024x1024.zip --mirror=1 \
--network=./training-runs_1024/00010-stylegan3-r-ffhq-1024x1024-gpus3-batch3-gamma32/network-snapshot-000000.pkl