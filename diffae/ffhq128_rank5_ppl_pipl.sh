#!/bin/bash
#SBATCH -p chaos
#SBATCH -A shared-mhug-staff
#SBATCH -t 23:59:00
#SBATCH --gres gpu:1
#SBATCH -o o_file/eval/ffhq128_rank5_ppl_pipl%j.o
#SBATCH -e e_file/eval/ffhq128_rank5_ppl_pipl%j.e

source /nfs/data_chaos/czhang/anaconda3/bin/activate
conda activate householdergan
export TMPDIR=/nfs/data_chaos/czhang/tmp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# LPIPS/VGG weights live on NFS to avoid home-disk quota issues
export TORCH_HOME=/nfs/data_chaos/czhang/torch_hub
export XDG_CACHE_HOME=/nfs/data_chaos/czhang/.cache
mkdir -p "$TORCH_HOME/checkpoints" "$XDG_CACHE_HOME"

RANK=5
CKPT=checkpoints/ffhq128_autoenc_rank5/checkpoints/epoch=56-step=124659.ckpt
FACTOR=factors/ffhq128_autoenc_rank${RANK}.pt
OUT=out_file/eval/ffhq128_rank${RANK}_ppl_pipl.out

mkdir -p factors evals out_file/eval o_file/eval e_file/eval

{
  echo "=== closed_form_factorization rank=${RANK} ==="
  python closed_form_factorization.py \
    --out ${FACTOR} \
    ${CKPT} \
    --is_ortho \
    --diag_size ${RANK}

  echo "=== PPL rank=${RANK} ==="
  python ppl.py \
    --ckpt ${CKPT} \
    --rank ${RANK} \
    --size 128 \
    --batch 16 \
    --n_sample 5000 \
    --eps 1e-4 \
    --sampling full

  echo "=== PIPL rank=${RANK} ==="
  python pipl.py \
    --ckpt ${CKPT} \
    --rank ${RANK} \
    --factor ${FACTOR} \
    --size 128 \
    --batch 16 \
    --n_sample 5000 \
    --eps 1e-4 \
    --sampling full

  echo "=== done ==="
} > ${OUT} 2>&1
