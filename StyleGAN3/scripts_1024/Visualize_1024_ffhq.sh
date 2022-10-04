#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH --gres gpu:1

eval "$(conda shell.bash hook)"
bash
conda activate stylegan3

python closed_form_factorization.py --out ./factor_1024.pt \
        --resume_pkl ./training-runs_1024/00007-stylegan3-r-ffhq-1024x1024-gpus4-batch4-gamma32/network-snapshot-000100.pkl \
        --is_ortho &

wait

python apply_factor.py --outdir=./visual_1024 --cfg=stylegan3-r  \
      --data=/nfs/data_chaos/datasets/FFHQ1024/ffhq-1024x1024.zip \
     --gpus=1 --batch-gpu=1 --batch=1 --gamma=32 --mbstd-group 1 \
    --resume=./training-runs_1024/00007-stylegan3-r-ffhq-1024x1024-gpus4-batch4-gamma32/network-snapshot-000100.pkl \
    --diag_size 10  --is_ortho True --factor ./factor_1024.pt

wait