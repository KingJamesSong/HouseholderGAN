#!/bin/bash
#SBATCH -p lambda
#SBATCH -A staff
#SBATCH --gres gpu:1

export PATH="/data/jzhang/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate eg3d

python closed_form_factorization.py --out ./factor_1024_SeFA_afhq.pt \
        --resume_pkl ./stylegan3-r-afhqv2-512x512.pkl

wait

python apply_factor.py --outdir=./visual_512_SeFA_afhq --cfg=stylegan3-r  \
      --data=/nfs/data_todi/ysong/AFHQ_v2/afhqv2-512x512.zip \
     --gpus=1 --batch-gpu=1 --batch=1 --gamma=16.4 --mbstd-group 1 \
    --resume=./stylegan3-r-afhqv2-512x512.pkl  \
    --diag_size 10  --is_ortho False --factor ./factor_1024_SeFA_afhq.pt

wait