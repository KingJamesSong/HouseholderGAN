#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH --gres gpu:1

eval "$(conda shell.bash hook)"
bash
conda activate pytorch3d

python -m torch.distributed.launch \
      --nproc_per_node=1 --master_port=9097 \
      train.py --batch 1 ./dataset/ffhq1024 --ckpt ./checkpoint_FFHQ/stylegan2-ffhq-config-f.pt --size 1024