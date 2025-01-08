#!/bin/bash
#SBATCH -p lambda
#SBATCH -A staff
#SBATCH -t 23:59:00
#SBATCH --gres gpu:1
#SBATCH -o o_file/eval/0108_horse128_HP.o
#SBATCH -e e_file/eval/0108_horse128_HP.e

source /nfs/data_chaos/czhang/anaconda3/bin/activate

conda activate householdergan
export TMPDIR=/nfs/data_chaos/czhang/tmp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
wandb login '61bbe1cdd46fd39ea897e6088bb2113126178cd8'


python run_horse128.py  > out_file/eval/0108_horse128_HP.out 
