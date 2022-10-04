#!/bin/bash
#SBATCH -p lambda
#SBATCH -A staff
#SBATCH --gres gpu:1

export PATH="/data/jzhang/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate eg3d

python compute_FID.py --cfg=stylegan3-r --gpus=1 --batch-gpu=1 --batch=1 --gamma=16.4 --mbstd-group 1 \
    --data=/nfs/data_todi/ysong/AFHQ_v2/afhqv2-512x512.zip \
    --resume=./training-runs_afhq_512/00000-stylegan3-r-afhqv2-512x512-gpus3-batch6-gamma16.4/network-snapshot-000120.pkl \
    --outdir=/nfs/data_chaos/jzhang/cvpr2023/visual_1024_AFHQ_FID