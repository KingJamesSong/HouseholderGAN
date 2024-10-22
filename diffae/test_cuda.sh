#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH -t 01:00:00
#SBATCH --gres gpu:1
#SBATCH -o o_file/train/test_cuda.o
#SBATCH -e e_file/train/test_cuda.e

source /nfs/data_chaos/czhang/anaconda3/bin/activate

conda activate householdergan
export TMPDIR=/nfs/data_chaos/czhang/tmp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python


python test_cuda.py  > out_file/train/test_cuda.out 