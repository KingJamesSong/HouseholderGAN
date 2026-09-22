#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH -t 23:59:00
#SBATCH --gres gpu:1
#SBATCH -o o_file/eval/0222_ffhq128_init_eval_%j.o
#SBATCH -e e_file/eval/0222_ffhq128_init_eval_%j.e

# Usage:
#   sbatch --export=INIT=svd_identity,STEP=0 ffhq128_init_ablation_eval.sh

source /nfs/data_chaos/czhang/anaconda3/bin/activate
conda activate householdergan
export TMPDIR=/nfs/data_chaos/czhang/tmp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TORCH_HOME=/nfs/data_chaos/czhang/torch_hub
export XDG_CACHE_HOME=/nfs/data_chaos/czhang/.cache
mkdir -p "$TORCH_HOME/checkpoints" "$XDG_CACHE_HOME"

cd /nfs/data_chaos/czhang/HouseholderGAN/diffae
mkdir -p out_file/eval o_file/eval e_file/eval

INIT=${INIT:?INIT required}
STEP=${STEP:?STEP required}

python run_ffhq128_init_ablation.py --mode eval --init ${INIT} --step ${STEP} --gpus 0 \
  >> out_file/eval/0222_ffhq128_init_${INIT}_step${STEP}.out 2>&1
