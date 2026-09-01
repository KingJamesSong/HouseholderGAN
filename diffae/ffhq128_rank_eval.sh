#!/bin/bash
#SBATCH -p chaos
#SBATCH -A shared-mhug-staff
#SBATCH -t 23:59:00
#SBATCH --gres gpu:1
#SBATCH -o o_file/eval/ffhq128_rank%j.o
#SBATCH -e e_file/eval/ffhq128_rank%j.e

# Usage:
#   sbatch --export=RANK=5 ffhq128_rank_eval.sh
#   sbatch --export=RANK=10 ffhq128_rank_eval.sh
#   sbatch --export=RANK=512 ffhq128_rank_eval.sh

source /nfs/data_chaos/czhang/anaconda3/bin/activate

conda activate householdergan
export TMPDIR=/nfs/data_chaos/czhang/tmp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

RANK=${RANK:-5}

python run_ffhq128_rank_ablation.py --rank ${RANK} --mode eval --gpus 0 \
    > out_file/eval/ffhq128_rank${RANK}.out 2>&1
