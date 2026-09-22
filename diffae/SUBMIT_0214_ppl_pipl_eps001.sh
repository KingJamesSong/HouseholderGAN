#!/bin/bash
# Re-run PPL + PIPL for 0214 step checkpoints with eps=0.01.
# Default steps: 0 10000 25000 50000
# Override: STEPS="0 50000" bash SUBMIT_0214_ppl_pipl_eps001.sh
set -e
cd /nfs/data_chaos/czhang/HouseholderGAN/diffae
STEPS=${STEPS:-"0 10000 25000 50000"}
EPS=0.01

echo "Existing step checkpoints:"
ls -lh checkpoints/0214_ffhq128_autoenc_130M_multi_projector/checkpoints/step=*.ckpt
echo "eps=${EPS}"

for STEP in $STEPS; do
  CKPT="checkpoints/0214_ffhq128_autoenc_130M_multi_projector/checkpoints/step=${STEP}.ckpt"
  if [[ ! -f "$CKPT" ]]; then
    echo "skip missing: $CKPT"
    continue
  fi
  J=$(sbatch --parsable --export=STEP=${STEP},EPS=${EPS} ffhq128_steps_ablation_ppl_pipl.sh)
  echo "submitted PPL/PIPL step=${STEP} eps=${EPS} -> job $J"
done

squeue -u "$USER" | head
