#!/bin/bash
#SBATCH -p lambda
#SBATCH -A staff
#SBATCH --gres gpu:1

export PATH="/data/jzhang/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate eg3d

python dataset_tool.py --source=/nfs/data_lambda/jzhang/github/siggraph/stylegan2-pytorch/dataset/ffhq1024_lmdb \
      --dest=/nfs/data_chaos/ysong/FFHQ1024/ffhq-1024x1024.zip \
    --resolution=1024x1024