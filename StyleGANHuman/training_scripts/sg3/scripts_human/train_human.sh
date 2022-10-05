#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH --gres gpu:2

eval "$(conda shell.bash hook)"
bash
conda activate stylegan3

python train.py --outdir=/nfs/data_chaos/jzhang/cvpr2023/stylehuman --cfg=stylegan3-r --gpus=1 --batch=2 --gamma=12.4 --mbstd-group 2 \
    --mirror=1 --aug=noaug --data=/nfs/data_chaos/jzhang/dataset/SHHQ-1.0/ --square=False --snap=5 \
    --resume=./stylegan_human_v3_512.pkl --diag_size 10  --is_ortho True