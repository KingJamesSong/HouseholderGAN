#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH -t 23:59:00
#SBATCH --gres gpu:1
#SBATCH -o o_file/1007_prepare_data.o
#SBATCH -e e_file/1007_prepare_data.e

source /nfs/data_chaos/czhang/anaconda3/bin/activate

conda activate householdergan
export TMPDIR=/nfs/data_chaos/czhang/tmp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
wandb login '61bbe1cdd46fd39ea897e6088bb2113126178cd8'


python -m torch.distributed.launch train_1024.py \
      --nproc_per_node=4 --master_port=9032 \
      train_1024.py --batch 8 '/nfs/datasets/FFHQ' \
      --ckpt '/nfs/data_chaos/czhang/HouseholderGAN/pretrained/FFHQ256_Pre-trained.pt' --size 1024 --ortho_id -2 --iter 10000000 \
      --checkpoints_dir 'ckpt/1007test/' \
      --sample_dir 'sample/1007test/' --loadd --training_FULL --diag_size 10 & > out_file/1007_prepare_data.out