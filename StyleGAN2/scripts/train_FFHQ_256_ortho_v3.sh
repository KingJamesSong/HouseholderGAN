#!/bin/bash
#SBATCH -p lambda
#SBATCH -A staff
#SBATCH --gres gpu:2

export PATH="/data/jzhang/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate styleganv2

python -m torch.distributed.launch \
      --nproc_per_node=2 --master_port=9094 \
      train_256.py --batch 2 ./data/ffhq256 --ckpt ./checkpoint_FFHQ/550000.pt --size 256 --is_ortho --iter 551000