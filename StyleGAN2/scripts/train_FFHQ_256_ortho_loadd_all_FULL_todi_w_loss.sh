#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH --gres gpu:1

eval "$(conda shell.bash hook)"
bash
conda activate eg3d

python -m torch.distributed.launch \
      --nproc_per_node=1 --master_port=9092 \
      train_256_w_loss.py --batch 8 ./data/ffhq256 \
      --ckpt ./checkpoint_FFHQ/550000.pt --size 256 --ortho_id -2 --iter 552000 \
      --checkpoints_dir checkpoint_FFHQ_loadd_all_FULL_w_loss_neighb \
      --sample_dir samplev_loadd_all_FULL_w_loss_neighb --loadd --training_FULL