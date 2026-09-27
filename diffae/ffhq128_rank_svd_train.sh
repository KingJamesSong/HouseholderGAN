#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH -t 23:59:00
#SBATCH --gres gpu:2
#SBATCH -o o_file/train/0225_ffhq128_rank_svd_%j.o
#SBATCH -e e_file/train/0225_ffhq128_rank_svd_%j.e

# Usage:
#   sbatch --export=RANK=5   ffhq128_rank_svd_train.sh
#   sbatch --export=RANK=10  ffhq128_rank_svd_train.sh
#   sbatch --export=RANK=512 ffhq128_rank_svd_train.sh

source /nfs/data_chaos/czhang/anaconda3/bin/activate
conda activate householdergan
export TMPDIR=/nfs/data_chaos/czhang/tmp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd /nfs/data_chaos/czhang/HouseholderGAN/diffae
mkdir -p out_file/train o_file/train e_file/train /nfs/data_chaos/czhang/tmp

RANK=${RANK:?RANK required}

python run_ffhq128_rank_svd_ablation.py --rank ${RANK} \
  >> out_file/train/0225_ffhq128_rank${RANK}_svd.out 2>&1
