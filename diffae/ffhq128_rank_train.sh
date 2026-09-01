#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH -t 23:59:00
#SBATCH --gres gpu:2
#SBATCH -o o_file/train/ffhq128_rank%j.o
#SBATCH -e e_file/train/ffhq128_rank%j.e

# Usage:
#   sbatch --export=RANK=5 ffhq128_rank_train.sh
#   sbatch --export=RANK=10 ffhq128_rank_train.sh
#   sbatch --export=RANK=512 ffhq128_rank_train.sh

source /nfs/data_chaos/czhang/anaconda3/bin/activate

conda activate householdergan
export TMPDIR=/nfs/data_chaos/czhang/tmp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

RANK=${RANK:-5}

python run_ffhq128_rank_ablation.py --rank ${RANK} --mode train \
    > out_file/train/ffhq128_rank${RANK}.out 2>&1
