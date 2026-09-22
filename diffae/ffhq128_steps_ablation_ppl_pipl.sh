#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH -t 23:59:00
#SBATCH --gres gpu:1
#SBATCH -o o_file/eval/0214_ffhq128_steps_ppl_pipl_%j.o
#SBATCH -e e_file/eval/0214_ffhq128_steps_ppl_pipl_%j.e

# Usage:
#   sbatch --export=STEP=0,EPS=0.01     ffhq128_steps_ablation_ppl_pipl.sh
#   sbatch --export=STEP=10000,EPS=1e-1 ffhq128_steps_ablation_ppl_pipl.sh
#
# Defaults: STEP required, EPS=1e-1

source /nfs/data_chaos/czhang/anaconda3/bin/activate
conda activate householdergan
export TMPDIR=/nfs/data_chaos/czhang/tmp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TORCH_HOME=/nfs/data_chaos/czhang/torch_hub
export XDG_CACHE_HOME=/nfs/data_chaos/czhang/.cache
mkdir -p "$TORCH_HOME/checkpoints" "$XDG_CACHE_HOME"

cd /nfs/data_chaos/czhang/HouseholderGAN/diffae
mkdir -p factors evals out_file/eval o_file/eval e_file/eval

STEP=${STEP:?STEP env var required}
EPS=${EPS:-1e-1}
# 0214 multi_projector uses diag_size=10
RANK=10
CKPT=checkpoints/0214_ffhq128_autoenc_130M_multi_projector/checkpoints/step=${STEP}.ckpt
FACTOR=factors/0214_ffhq128_autoenc_130M_multi_projector_step${STEP}.pt
# tag results by eps so re-runs do not overwrite older files
EPS_TAG=$(python -c "print(f'{float(\"${EPS}\"):g}'.replace('.', 'p'))")
OUT_NAME=0214_ffhq128_autoenc_130M_multi_projector_step${STEP}_eps${EPS_TAG}
OUT=out_file/eval/0214_ffhq128_steps_step${STEP}_ppl_pipl_eps${EPS_TAG}.out

if [[ ! -f "$CKPT" ]]; then
  echo "missing ckpt: $CKPT" >&2
  exit 1
fi

{
  if [[ ! -f "$FACTOR" ]]; then
    echo "=== closed_form_factorization step=${STEP} ==="
    python closed_form_factorization.py \
      --out ${FACTOR} \
      ${CKPT} \
      --is_ortho \
      --diag_size ${RANK}
  else
    echo "=== reuse factor ${FACTOR} ==="
  fi

  echo "=== PPL step=${STEP} eps=${EPS} ==="
  python ppl.py \
    --ckpt ${CKPT} \
    --rank ${RANK} \
    --out_name ${OUT_NAME} \
    --size 128 \
    --batch 16 \
    --n_sample 5000 \
    --eps ${EPS} \
    --sampling full

  echo "=== PIPL step=${STEP} eps=${EPS} ==="
  python pipl.py \
    --ckpt ${CKPT} \
    --rank ${RANK} \
    --out_name ${OUT_NAME} \
    --factor ${FACTOR} \
    --size 128 \
    --batch 16 \
    --n_sample 5000 \
    --eps ${EPS} \
    --sampling full

  echo "=== done step=${STEP} eps=${EPS} ==="
} > ${OUT} 2>&1
