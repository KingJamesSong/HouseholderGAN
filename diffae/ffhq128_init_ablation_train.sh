#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH -t 23:59:00
#SBATCH --gres gpu:2
#SBATCH -o o_file/train/0222_ffhq128_init_ablation_%j.o
#SBATCH -e e_file/train/0222_ffhq128_init_ablation_%j.e

# Initialization ablation train.
# Usage:
#   sbatch --export=INIT=svd_identity ffhq128_init_ablation_train.sh
#   sbatch --export=INIT=random      ffhq128_init_ablation_train.sh

source /nfs/data_chaos/czhang/anaconda3/bin/activate
conda activate householdergan
export TMPDIR=/nfs/data_chaos/czhang/tmp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd /nfs/data_chaos/czhang/HouseholderGAN/diffae
mkdir -p out_file/train o_file/train e_file/train /nfs/data_chaos/czhang/tmp

INIT=${INIT:-svd_identity}

python run_ffhq128_init_ablation.py --init ${INIT} \
  >> out_file/train/0222_ffhq128_init_${INIT}.out 2>&1
