#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH --gres gpu:1
#SBATCH -o /data/ysong/vp_test1.txt

eval "$(conda shell.bash hook)"
bash

cd /nfs/data_chaos/ysong/StyleGAN2/

conda activate ood

python -m torch.distributed.launch \
      --nproc_per_node=1 --master_port=9091 \
      train_256.py --batch 4 /nfs/data_lambda/jzhang/github/siggraph/stylegan2-pytorch/data/ffhq256 \
      --ckpt ./checkpoint_FFHQ/550000.pt --size 256 --ortho_id -2 --iter 552000 \
      --checkpoints_dir checkpoint_FFHQ_loadd_all_FULL_wo_loss_b4 \
      --sample_dir samplev_loadd_all_FULL_wo_loss_b4 --loadd --training_FULL --diag_size 10 &

wait
