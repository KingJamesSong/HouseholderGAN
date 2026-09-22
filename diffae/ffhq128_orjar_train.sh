#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH -t 23:59:00
#SBATCH --gres gpu:2
#SBATCH -o o_file/train/0125_ffhq128_multi_mlp_OrJaR_%j.o
#SBATCH -e e_file/train/0125_ffhq128_multi_mlp_OrJaR_%j.e

source /nfs/data_chaos/czhang/anaconda3/bin/activate
conda activate householdergan
export TMPDIR=/nfs/data_chaos/czhang/tmp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd /nfs/data_chaos/czhang/HouseholderGAN/diffae
mkdir -p out_file/train o_file/train e_file/train /nfs/data_chaos/czhang/tmp

python run_ffhq128_orjar.py > out_file/train/0125_ffhq128_multi_mlp_OrJaR.out 2>&1
