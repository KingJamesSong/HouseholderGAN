#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH -t 23:59:00
#SBATCH --gres gpu:2
#SBATCH -o o_file/train/0927_ffhq128_proj_ablation_%j.o
#SBATCH -e e_file/train/0927_ffhq128_proj_ablation_%j.e

# Usage:
#   sbatch --export=METHOD=householder,RANK=10  ffhq128_projector_ablation_train.sh
#   sbatch --export=METHOD=householder,RANK=512 ffhq128_projector_ablation_train.sh
#   sbatch --export=METHOD=lora,RANK=10         ffhq128_projector_ablation_train.sh

source /nfs/data_chaos/czhang/anaconda3/bin/activate
conda activate householdergan
export TMPDIR=/nfs/data_chaos/czhang/tmp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd /nfs/data_chaos/czhang/HouseholderGAN/diffae
mkdir -p out_file/train o_file/train e_file/train /nfs/data_chaos/czhang/tmp

METHOD=${METHOD:?METHOD required}
RANK=${RANK:?RANK required}

python run_ffhq128_projector_ablation.py --method ${METHOD} --rank ${RANK} \
  >> out_file/train/0927_ffhq128_${METHOD}_rank${RANK}.out 2>&1
