#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH --gres gpu:3

eval "$(conda shell.bash hook)"
bash
conda activate eg3d

python -m torch.distributed.launch \
      --nproc_per_node=3 --master_port=9094 \
      train_256.py --batch 32 ./data/ffhq256 \
      --ckpt ./checkpoint_FFHQ/550000.pt --size 256 --ortho_id -2 --iter 552000 \
      --checkpoints_dir checkpoint_FFHQ_loadd_all_FULL_wo_loss_b32 \
      --sample_dir samplev_loadd_all_FULL_wo_loss_b32 --loadd --training_FULL --diag_size 10 &

wait
