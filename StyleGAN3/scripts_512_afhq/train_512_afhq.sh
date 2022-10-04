#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH --gres gpu:3

eval "$(conda shell.bash hook)"
bash
conda activate stylegan3

python train.py --outdir=./training-runs_afhq_512 --cfg=stylegan3-r \
          --data=/nfs/data_todi/ysong/AFHQ_v2/afhqv2-512x512.zip \
      	--cfg=stylegan3-r --gpus=3 --batch-gpu=2 --batch=6 --gamma=16.4 --mbstd-group 2 \
      	--resume=./stylegan3-r-afhqv2-512x512.pkl \
        --diag_size 10  --is_ortho True --snap 5