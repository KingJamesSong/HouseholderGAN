#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH -t 23:59:00
#SBATCH --gres gpu:1
#SBATCH -o o_file/eval/0927_proj_fid_%j.o
#SBATCH -e e_file/eval/0927_proj_fid_%j.e

# Usage:
#   sbatch --export=METHOD=householder,RANK=10,STEP=20000 ffhq128_projector_ablation_fid.sh

source /nfs/data_chaos/czhang/anaconda3/bin/activate
conda activate householdergan
export TMPDIR=/nfs/data_chaos/czhang/tmp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TORCH_HOME=/nfs/data_chaos/czhang/torch_hub
export XDG_CACHE_HOME=/nfs/data_chaos/czhang/.cache
mkdir -p "$TORCH_HOME/checkpoints" "$XDG_CACHE_HOME"

cd /nfs/data_chaos/czhang/HouseholderGAN/diffae
mkdir -p out_file/eval o_file/eval e_file/eval evals

METHOD=${METHOD:?}
RANK=${RANK:?}
STEP=${STEP:?}

python run_ffhq128_projector_ablation.py \
  --mode eval --method ${METHOD} --rank ${RANK} --step ${STEP} --gpus 0 \
  >> out_file/eval/0927_${METHOD}_rank${RANK}_step${STEP}_fid.out 2>&1
