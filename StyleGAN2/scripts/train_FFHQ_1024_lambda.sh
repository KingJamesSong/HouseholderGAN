#!/bin/bash
#SBATCH -p lambda
#SBATCH -A staff
#SBATCH --gres gpu:2

export PATH="/data/jzhang/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate styleganv2

python -m torch.distributed.launch \
      --nproc_per_node=1 --master_port=9097 \
      train.py --batch 1 ./dataset/ffhq1024 --ckpt ./checkpoint_FFHQ/stylegan2-ffhq-config-f.pt --size 1024