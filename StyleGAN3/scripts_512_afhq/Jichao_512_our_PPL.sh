#!/bin/bash
#SBATCH -p lambda
#SBATCH -A staff
#SBATCH --gres gpu:1

export PATH="/data/jzhang/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate eg3d

python apply_ppl.py --outdir=/nfs/data_chaos/jzhang/cvpr2023/PPL --cfg=stylegan3-r  \
      --data=/nfs/data_todi/ysong/AFHQ_v2/afhqv2-512x512.zip \
     --gpus=1 --batch-gpu=1 --batch=1 --gamma=16.4 --mbstd-group 1 \
    --resume=/nfs/data_todi/ysong/training-runs_afhq_512/00000-stylegan3-r-afhqv2-512x512-gpus3-batch6-gamma16.4/network-snapshot-000120.pkl \
    --diag_size 10  --is_ortho True --factor ./factor_1024_SeFA_afhq.pt

wait