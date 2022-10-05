#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH --gres gpu:1

eval "$(conda shell.bash hook)"
bash
conda activate stylegan3

python closed_form_factorization.py --out ./factor_256_sefa.pt \
        --resume_pkl ./stylegan3-r-ffhqu-256x256.pkl &

wait

python apply_factor.py --outdir=./training-runs_256 --cfg=stylegan3-r  --data=./ffhq-256x256.zip \
     --gpus=1 --batch=1 --gamma=1 --mirror=1 --aug=noaug --cbase=16384 --dlr=0.0025 --mbstd-group 1 \
    --resume=./stylegan3-r-ffhqu-256x256.pkl --diag_size 10  --is_ortho False --factor ./factor_256_sefa.pt

wait