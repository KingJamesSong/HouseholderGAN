#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH -t 23:59:00
#SBATCH --gres gpu:2
#SBATCH -o o_file/train/0214_ffhq128_steps_ablation_%j.o
#SBATCH -e e_file/train/0214_ffhq128_steps_ablation_%j.e

# Finetuning-steps ablation for multi_projector:
#   saves checkpoints at 0 / 10k / 25k / 50k / 100k under
#   checkpoints/0214_ffhq128_autoenc_130M_multi_projector/checkpoints/
#
# Usage:
#   sbatch ffhq128_steps_ablation_train.sh
# Optional auto-resume if walltime cuts off before 100k:
#   J=$(sbatch --parsable ffhq128_steps_ablation_train.sh)
#   sbatch --dependency=afterany:$J ffhq128_steps_ablation_train.sh

source /nfs/data_chaos/czhang/anaconda3/bin/activate
conda activate householdergan
export TMPDIR=/nfs/data_chaos/czhang/tmp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd /nfs/data_chaos/czhang/HouseholderGAN/diffae
mkdir -p out_file/train o_file/train e_file/train /nfs/data_chaos/czhang/tmp checkpoints/0214_ffhq128_autoenc_130M_multi_projector/checkpoints

# append so resume runs keep prior logs
python run_ffhq128_steps_ablation.py \
  >> out_file/train/0214_ffhq128_steps_ablation.out 2>&1
