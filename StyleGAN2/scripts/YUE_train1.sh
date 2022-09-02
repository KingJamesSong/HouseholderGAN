#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH --gres gpu:1
#SBATCH -o /data/ysong/vp_test1.txt

eval "$(conda shell.bash hook)"
bash

cd /nfs/data_chaos/ysong/StyleGAN2/

conda activate eg3d

python -m torch.distributed.launch \
      --nproc_per_node=1 --master_port=9091 \
      train_256.py --batch 8 /nfs/data_lambda/jzhang/github/siggraph/stylegan2-pytorch/data/ffhq256 \
      --ckpt ./checkpoint_FFHQ/550000.pt --size 256 --ortho_id -3 --iter 552000 \
      --checkpoints_dir checkpoint_FFHQ_lr0.01_b8_id3 \
      --sample_dir samplev_lr0.01_b8_id3 --loadd --training_FULL --diag_size 10 --lr_ortho 0.01 \

wait

python closed_form_factorization.py --out ./factor/factor_b8_2000_lr001_id3.pt ./checkpoint_FFHQ_lr0.01_b8_id3/552000.pt --is_ortho --diag_size 10 \

wait

python ppl_sefa.py ./checkpoint_FFHQ_lr0.01_b8_id3/552000.pt --factor ./factor/factor_b8_2000_lr001_id3.pt --ortho_id -3 --sampling full --eps 1.0 \

wait

python ppl.py ./checkpoint_FFHQ_lr0.01_b8_id3/552000.pt --ortho_id -3 --sampling full \

wait

python test_VP_pairs.py --ortho_id -3 --size 256 \
        --ckpt ./checkpoint_FFHQ_lr0.01_b8_id3/552000.pt \
        --factor ./factor/factor_b8_2000_lr001_id3.pt --output_dir output_vp_pairsb8_001 &

wait