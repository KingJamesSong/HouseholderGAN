#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH --gres gpu:1

eval "$(conda shell.bash hook)"
bash
conda activate stylegan3

python closed_form_factorization.py --out ./factor_1024_human.pt \
        --resume_pkl ./training-runs_afhq_512/00000-stylegan3-r-afhqv2-512x512-gpus3-batch6-gamma16.4/network-snapshot-000120.pkl \
        --is_ortho &

wait

python apply_factor.py --outdir=./visual_1024_afhq --cfg=stylegan3-r  \
      --data=/nfs/data_todi/ysong/AFHQ_v2/afhqv2-512x512.zip \
     --gpus=1 --batch-gpu=1 --batch=1 --gamma=16.4 --mbstd-group 1 \
    --resume=./training-runs_afhq_512/00000-stylegan3-r-afhqv2-512x512-gpus3-batch6-gamma16.4/network-snapshot-000120.pkl \
    --diag_size 10  --is_ortho True --factor ./factor_1024_afhq.pt

wait