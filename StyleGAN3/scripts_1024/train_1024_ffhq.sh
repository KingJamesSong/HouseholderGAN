#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH --gres gpu:3

eval "$(conda shell.bash hook)"
bash
conda activate stylegan3

python train.py --outdir=./training-runs_1024 --cfg=stylegan3-r \
          --data=/nfs/data_chaos/datasets/FFHQ1024/ffhq-1024x1024.zip \
      	--cfg=stylegan3-r --gpus=3 --batch-gpu=1 --batch=3 --gamma=32 --mbstd-group 1 \
      	--resume=./training-runs_1024/00009-stylegan3-r-ffhq-1024x1024-gpus3-batch3-gamma32/network-snapshot-000060.pkl \
        --diag_size 10  --is_ortho True --snap 5