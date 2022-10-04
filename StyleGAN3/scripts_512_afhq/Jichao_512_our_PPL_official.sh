#!/bin/bash
#SBATCH -p lambda
#SBATCH -A staff
#SBATCH --gres gpu:1

export PATH="/data/jzhang/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate eg3d

python calc_metrics.py --metrics=ppl2_wend \
    --network=/nfs/data_todi/ysong/training-runs_afhq_512/00000-stylegan3-r-afhqv2-512x512-gpus3-batch6-gamma16.4/network-snapshot-000120.pkl