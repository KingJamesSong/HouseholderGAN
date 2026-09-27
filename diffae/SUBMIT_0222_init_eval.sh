#!/bin/bash
# Submit FID + PPL/PIPL for 0222 init ablation checkpoints.
# Default: INIT=svd_identity, all existing step=*.ckpt, EPS=0.01
#
# Usage:
#   bash SUBMIT_0222_init_eval.sh
#   STEPS="0 10000" EPS=0.01 bash SUBMIT_0222_init_eval.sh
set -e
cd /nfs/data_chaos/czhang/HouseholderGAN/diffae

INIT=${INIT:-svd_identity}
EPS=${EPS:-0.01}
EXP=0222_ffhq128_autoenc_130M_multi_projector_init_${INIT}
CKPT_DIR=checkpoints/${EXP}/checkpoints

echo "=== ${EXP} ==="
ls -lh ${CKPT_DIR}/step=*.ckpt

if [[ -z "${STEPS:-}" ]]; then
  STEPS=""
  for f in ${CKPT_DIR}/step=*.ckpt; do
    [[ -f "$f" ]] || continue
    base=$(basename "$f")
    [[ "$base" == *_weights.ckpt ]] && continue
    step=${base#step=}
    step=${step%.ckpt}
    STEPS="${STEPS} ${step}"
  done
fi
STEPS=$(echo $STEPS)
echo "STEPS=${STEPS}"
echo "EPS=${EPS}"

for STEP in $STEPS; do
  CKPT="${CKPT_DIR}/step=${STEP}.ckpt"
  if [[ ! -f "$CKPT" ]]; then
    echo "skip missing: $CKPT"
    continue
  fi
  J1=$(sbatch --parsable --export=INIT=${INIT},STEP=${STEP} ffhq128_init_ablation_eval.sh)
  echo "submitted FID  init=${INIT} step=${STEP} -> job $J1"
  J2=$(sbatch --parsable --export=INIT=${INIT},STEP=${STEP},EPS=${EPS} ffhq128_init_ablation_ppl_pipl.sh)
  echo "submitted PPL/PIPL init=${INIT} step=${STEP} eps=${EPS} -> job $J2"
done

squeue -u "$USER" | head
