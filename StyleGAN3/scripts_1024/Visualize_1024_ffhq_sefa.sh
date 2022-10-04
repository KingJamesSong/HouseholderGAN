#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH --gres gpu:1

eval "$(conda shell.bash hook)"
bash
conda activate stylegan3

python closed_form_factorization.py --out ./factor_1024_sefa.pt \
        --resume_pkl ./stylegan3-r-ffhq-1024x1024.pkl &

wait

python apply_factor.py --outdir=./visual_1024_sefa --cfg=stylegan3-r  \
      --data=/nfs/data_chaos/datasets/FFHQ1024/ffhq-1024x1024.zip \
     --gpus=1 --batch-gpu=1 --batch=1 --gamma=32 --mbstd-group 1 \
    --resume=./stylegan3-r-ffhq-1024x1024.pkl \
    --diag_size 10  --is_ortho False --factor ./factor_1024_sefa.pt

wait