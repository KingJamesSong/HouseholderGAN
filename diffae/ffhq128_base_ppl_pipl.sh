#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH -t 23:59:00
#SBATCH --gres gpu:1
#SBATCH -o o_file/eval/ffhq128_base_ppl_pipl_%j.o
#SBATCH -e e_file/eval/ffhq128_base_ppl_pipl_%j.e

source /nfs/data_chaos/czhang/anaconda3/bin/activate
conda activate householdergan
export TMPDIR=/nfs/data_chaos/czhang/tmp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TORCH_HOME=/nfs/data_chaos/czhang/torch_hub
export XDG_CACHE_HOME=/nfs/data_chaos/czhang/.cache
mkdir -p "$TORCH_HOME/checkpoints" "$XDG_CACHE_HOME"

cd /nfs/data_chaos/czhang/HouseholderGAN/diffae
mkdir -p factors evals out_file/eval o_file/eval e_file/eval

CKPT=checkpoints/ffhq128_autoenc_130M/last.ckpt
FACTOR=factors/ffhq128_autoenc_130M.pt
OUT_NAME=ffhq128_autoenc_130M
OUT=out_file/eval/ffhq128_autoenc_130M_ppl_pipl.out

{
  echo "=== closed_form_factorization (vanilla DiffAE) ==="
  python closed_form_factorization.py \
    --out ${FACTOR} \
    ${CKPT}

  echo "=== PPL vanilla DiffAE ==="
  python ppl.py \
    --ckpt ${CKPT} \
    --vanilla \
    --out_name ${OUT_NAME} \
    --size 128 \
    --batch 16 \
    --n_sample 5000 \
    --eps 1e-1 \
    --sampling full

  echo "=== PIPL vanilla DiffAE ==="
  python pipl.py \
    --ckpt ${CKPT} \
    --vanilla \
    --out_name ${OUT_NAME} \
    --factor ${FACTOR} \
    --size 128 \
    --batch 16 \
    --n_sample 5000 \
    --eps 1e-1 \
    --sampling full

  echo "=== done ==="
} > ${OUT} 2>&1
