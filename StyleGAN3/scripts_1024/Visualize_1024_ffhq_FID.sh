#!/bin/bash
#SBATCH -p lambda
#SBATCH -A staff
#SBATCH --gres gpu:1

export PATH="/data/jzhang/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate eg3d

python compute_FID.py --cfg=stylegan3-r --gpus=1 --batch-gpu=1 --batch=1 --gamma=32 --mbstd-group 1 \
    --data=/nfs/data_chaos/datasets/FFHQ1024/ffhq-1024x1024.zip \
    --resume=./training-runs_1024/00010-stylegan3-r-ffhq-1024x1024-gpus3-batch3-gamma32/network-snapshot-000060.pkl \
    --outdir=/nfs/data_chaos/jzhang/cvpr2023/visual_1024_FFHQ_FID_00010