#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH -t 23:59:00
#SBATCH --gres gpu:1
#SBATCH -o o_file/eval/0214_ffhq128_steps_eval_%j.o
#SBATCH -e e_file/eval/0214_ffhq128_steps_eval_%j.e

# Usage:
#   sbatch --export=STEP=0     ffhq128_steps_ablation_eval.sh
#   sbatch --export=STEP=10000 ffhq128_steps_ablation_eval.sh
#   sbatch --export=STEP=25000 ffhq128_steps_ablation_eval.sh

source /nfs/data_chaos/czhang/anaconda3/bin/activate
conda activate householdergan
export TMPDIR=/nfs/data_chaos/czhang/tmp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TORCH_HOME=/nfs/data_chaos/czhang/torch_hub

cd /nfs/data_chaos/czhang/HouseholderGAN/diffae
mkdir -p out_file/eval o_file/eval e_file/eval /nfs/data_chaos/czhang/tmp

STEP=${STEP:?STEP env var required}

python run_ffhq128_steps_ablation.py --mode eval --step ${STEP} --gpus 0 \
  >> out_file/eval/0214_ffhq128_steps_step${STEP}.out 2>&1
