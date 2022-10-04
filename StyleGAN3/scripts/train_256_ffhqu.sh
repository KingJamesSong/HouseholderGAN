#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH --gres gpu:4

eval "$(conda shell.bash hook)"
bash
conda activate stylegan3

python train.py --outdir=./training-runs_256 --cfg=stylegan3-r --data=./ffhq-256x256.zip \
     --gpus=3 --batch=6 --gamma=1 --mirror=1 --aug=noaug --cbase=16384 --dlr=0.0025 --mbstd-group 2 \
    --resume=./stylegan3-r-ffhqu-256x256.pkl --diag_size 10  --is_ortho True --snap 5