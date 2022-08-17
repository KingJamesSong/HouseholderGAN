#!/bin/bash
#SBATCH -p lambda
#SBATCH -A staff
#SBATCH --gres gpu:2

export PATH="/data/jzhang/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash
conda activate styleganv2

python -m torch.distributed.launch \
      --nproc_per_node=1 --master_port=9095 \
      train_256.py --batch 2 ./data/ffhq256 \
      --ckpt ./checkpoint_FFHQ/550000.pt --size 256 --ortho_id -2 --iter 560000 \
      --checkpoints_dir checkpoint_FFHQ_woloadd_all_partial --sample_dir samplev_woloadd_all_partial