#!/bin/bash
#SBATCH -p lambda
#SBATCH -A staff
#SBATCH --gres gpu:1

export PATH="/data/jzhang/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate eg3d

python apply_ppl_sefa.py --outdir=./visual_1024_ppl_sefa --cfg=stylegan3-r  \
      --data=/nfs/data_chaos/datasets/FFHQ1024/ffhq-1024x1024.zip \
     --gpus=1 --batch-gpu=1 --batch=1 --gamma=32 --mbstd-group 1 \
    --resume=./stylegan3-r-ffhq-1024x1024.pkl \
    --diag_size 10  --is_ortho False --factor ./factor_1024_sefa.pt

wait