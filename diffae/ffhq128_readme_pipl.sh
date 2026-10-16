#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH -t 12:00:00
#SBATCH --gres gpu:1
#SBATCH -o o_file/eval/0927_readme_pipl_%j.o
#SBATCH -e e_file/eval/0927_readme_pipl_%j.e

# Usage:
#   sbatch --export=ARM=householder5 ffhq128_readme_pipl.sh
#   ARM in {householder5,householder10,householder512,lora10,vanilla}

source /nfs/data_chaos/czhang/anaconda3/bin/activate
conda activate householdergan
export TMPDIR=/nfs/data_chaos/czhang/tmp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TORCH_HOME=/nfs/data_chaos/czhang/torch_hub
export XDG_CACHE_HOME=/nfs/data_chaos/czhang/.cache
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
mkdir -p "$TORCH_HOME/checkpoints" "$XDG_CACHE_HOME"

cd /nfs/data_chaos/czhang/HouseholderGAN/diffae
mkdir -p evals/readme_pipl out_file/eval o_file/eval e_file/eval

ARM=${ARM:?}
EPS=${EPS:-1e-4}
SAMPLES=${SAMPLES:-2000}

python -u evaluate_ffhq128_pipl_readme.py \
  --arm ${ARM} \
  --samples ${SAMPLES} \
  --batch-size 16 \
  --epsilon ${EPS} \
  --ddim-steps 100 \
  --pipl-directions 10 \
  --seed 0 \
  >> out_file/eval/readme_pipl_${ARM}_eps${EPS}_n${SAMPLES}.out 2>&1
