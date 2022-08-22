#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH --gres gpu:1
#SBATCH -o /data/ysong/vp_test.txt

eval "$(conda shell.bash hook)"
bash

cd /nfs/data_chaos/ysong/StyleGAN2/

conda activate ood


python closed_form_factorization.py --out ./factor/factor_b8_10000.pt ./checkpoint_FFHQ_loadd_all_FULL_wo_loss_b8_iter560000/560000.pt --is_ortho \

wait

python ppl.py ./checkpoint_FFHQ_loadd_all_FULL_wo_loss_b8_iter560000/560000.pt --ortho_id -2 --sampling full \

wait

python test_VP_pairs.py --ortho_id -2 --size 256 \
        --ckpt ./checkpoint_FFHQ_loadd_all_FULL_wo_loss_b8_iter560000/560000.pt  \
        --factor ./factor/factor_b8_10000.pt --output_dir output_vp_pairsb8_10000 &

wait