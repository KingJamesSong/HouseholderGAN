#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH --gres gpu:1

eval "$(conda shell.bash hook)"
bash
conda activate stylegan3

python apply_factor.py --outdir=./training-runs_256 --cfg=stylegan3-r \
        --data=./ffhq-256x256.zip \
        --gpus=1 --batch=1 --gamma=1 --mirror=1 --aug=noaug \
        --cbase=16384 --dlr=0.0025 --mbstd-group 1 \
        --resume=./training-runs_256/00003-stylegan3-r-ffhq-256x256-gpus3-batch6-gamma1/network-snapshot-000040.pkl \
        --diag_size 10  --is_ortho True --factor ./factor_256.pt