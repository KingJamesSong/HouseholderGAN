#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH -t 04:00:00
#SBATCH --gres gpu:1
#SBATCH -o o_file/eval/0927_proj_traverse_%j.o
#SBATCH -e e_file/eval/0927_proj_traverse_%j.e

# Usage:
#   sbatch --export=METHOD=householder,RANK=512,STEP=50000 \
#     ffhq128_projector_ablation_traverse.sh

source /nfs/data_chaos/czhang/anaconda3/bin/activate
conda activate householdergan
export TMPDIR=/nfs/data_chaos/czhang/tmp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TORCH_HOME=/nfs/data_chaos/czhang/torch_hub
export XDG_CACHE_HOME=/nfs/data_chaos/czhang/.cache
mkdir -p "$TORCH_HOME/checkpoints" "$XDG_CACHE_HOME"

cd /nfs/data_chaos/czhang/HouseholderGAN/diffae
mkdir -p factors out_file/eval o_file/eval e_file/eval

METHOD=${METHOD:?}
RANK=${RANK:?}
STEP=${STEP:?}
DEGREE=${DEGREE:-5}
N_SAMPLE=${N_SAMPLE:-7}
N_STEPS=${N_STEPS:-4}
T=${T:-20}

if [[ "$METHOD" == "householder" ]]; then
  EXP=0927_ffhq128_autoenc_rank${RANK}_svd_random
  FACTOR_FLAGS=(--is_ortho --diag_size ${RANK})
elif [[ "$METHOD" == "lora" ]]; then
  EXP=0927_ffhq128_autoenc_lora${RANK}_random
  FACTOR_FLAGS=(--is_lora --diag_size ${RANK})
else
  echo "unknown METHOD=$METHOD" >&2
  exit 1
fi

CKPT=checkpoints/${EXP}/checkpoints/step=${STEP}.ckpt
FACTOR=factors/${EXP}_step${STEP}.pt
OUT=out_file/eval/${EXP}_step${STEP}_traverse.out

if [[ ! -f "$CKPT" ]]; then
  echo "missing ckpt: $CKPT" >&2
  exit 1
fi

{
  if [[ ! -f "$FACTOR" ]]; then
    echo "=== closed_form_factorization ==="
    python closed_form_factorization.py \
      --out ${FACTOR} \
      ${CKPT} \
      "${FACTOR_FLAGS[@]}"
  else
    echo "=== reuse factor ${FACTOR} ==="
  fi

  echo "=== traverse method=${METHOD} rank=${RANK} step=${STEP} ==="
  python run_ffhq128_projector_ablation_traverse.py \
    --method ${METHOD} --rank ${RANK} --step ${STEP} \
    --factor ${FACTOR} \
    --n_sample ${N_SAMPLE} --n_steps ${N_STEPS} \
    --degree ${DEGREE} --T ${T}

  echo "=== done ==="
} > ${OUT} 2>&1
