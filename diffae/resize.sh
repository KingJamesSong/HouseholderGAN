#!/bin/bash
#SBATCH -p lambda
#SBATCH -A staff
#SBATCH -t 23:59:00
#SBATCH --gres gpu:1
#SBATCH -o o_file/train/1231_resize_horse_256.o
#SBATCH -e e_file/train/1231_resize_horse_256.e

source /nfs/data_chaos/czhang/anaconda3/bin/activate

conda activate householdergan
export TMPDIR=/nfs/data_chaos/czhang/tmp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
wandb login '61bbe1cdd46fd39ea897e6088bb2113126178cd8'


python data_resize_horse.py  > out_file/train/1231_resize_horse_256.out 
