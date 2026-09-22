#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH -t 23:59:00
#SBATCH --gres gpu:1
#SBATCH -o o_file/eval/ffhq128_rank_pipl_fair_%j.o
#SBATCH -e e_file/eval/ffhq128_rank_pipl_fair_%j.e

# Fair cross-rank PIPL: always sample among top-N_EIG singular vectors.
# Usage:
#   sbatch --export=RANK=5   ffhq128_rank_pipl_fair.sh
#   sbatch --export=RANK=10  ffhq128_rank_pipl_fair.sh
#   sbatch --export=RANK=512 ffhq128_rank_pipl_fair.sh

source /nfs/data_chaos/czhang/anaconda3/bin/activate
conda activate householdergan
export TMPDIR=/nfs/data_chaos/czhang/tmp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TORCH_HOME=/nfs/data_chaos/czhang/torch_hub
export XDG_CACHE_HOME=/nfs/data_chaos/czhang/.cache
mkdir -p "$TORCH_HOME/checkpoints" "$XDG_CACHE_HOME"

cd /nfs/data_chaos/czhang/HouseholderGAN/diffae
mkdir -p factors evals out_file/eval o_file/eval e_file/eval

RANK=${RANK:?RANK env var required}
N_EIG=${N_EIG:-5}

case "${RANK}" in
  5)
    CKPT=checkpoints/ffhq128_autoenc_rank5/checkpoints/epoch=56-step=124659.ckpt
    ;;
  10)
    CKPT=checkpoints/ffhq128_autoenc_rank10/checkpoints/epoch=128-step=281250.ckpt
    ;;
  512)
    CKPT=checkpoints/ffhq128_autoenc_rank512/checkpoints/epoch=191-step=418750.ckpt
    ;;
  *)
    echo "unsupported RANK=${RANK}" >&2
    exit 1
    ;;
esac

FACTOR=factors/ffhq128_autoenc_rank${RANK}.pt
OUT_NAME=ffhq128_autoenc_rank${RANK}_fair
OUT=out_file/eval/ffhq128_rank${RANK}_pipl_fair.out

if [[ ! -f "$CKPT" ]]; then
  echo "missing ckpt: $CKPT" >&2
  exit 1
fi

{
  echo "=== closed_form_factorization rank=${RANK} ==="
  python closed_form_factorization.py \
    --out ${FACTOR} \
    ${CKPT} \
    --is_ortho \
    --diag_size ${RANK}

  echo "=== PIPL fair rank=${RANK} n_eig=${N_EIG} ==="
  python pipl.py \
    --ckpt ${CKPT} \
    --rank ${RANK} \
    --n_eig ${N_EIG} \
    --out_name ${OUT_NAME} \
    --factor ${FACTOR} \
    --size 128 \
    --batch 16 \
    --n_sample 5000 \
    --eps 1e-1 \
    --sampling full

  echo "=== done rank=${RANK} ==="
} > ${OUT} 2>&1
