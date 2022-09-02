#!/bin/bash
#SBATCH -p chaos
#SBATCH -A shared-mhug-staff
#SBATCH --gres gpu:1
#SBATCH -o /data/ysong/vp2.txt

eval "$(conda shell.bash hook)"
bash

cd /nfs/data_chaos/ysong/StyleGAN2/

conda activate eg3d


python fid.py ./checkpoint_FFHQ_b40/552000.pt --ortho_id -2 --inception inception_ffhq.pkl \

wait

python closed_form_factorization.py --out ./factor/factor_b40.pt ./checkpoint_FFHQ_b40/552000.pt --is_ortho --diag_size 10 \

wait

python ppl_sefa.py ./checkpoint_FFHQ_b40/552000.pt --factor ./factor/factor_b40.pt --ortho_id -2 --sampling full --eps 1.0 \

wait

python ppl.py ./checkpoint_FFHQ_b40/552000.pt --ortho_id -2 --sampling full &

wait