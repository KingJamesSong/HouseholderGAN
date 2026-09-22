#!/bin/bash
# Eval FID for existing 0214 step checkpoints (0 / 10k / 25k by default).
# Override: STEPS="0 10000 25000 50000" bash SUBMIT_0214_eval.sh
set -e
cd /nfs/data_chaos/czhang/HouseholderGAN/diffae
STEPS=${STEPS:-"0 10000 25000"}

echo "Existing step checkpoints:"
ls -lh checkpoints/0214_ffhq128_autoenc_130M_multi_projector/checkpoints/step=*.ckpt

for STEP in $STEPS; do
  CKPT="checkpoints/0214_ffhq128_autoenc_130M_multi_projector/checkpoints/step=${STEP}.ckpt"
  if [[ ! -f "$CKPT" ]]; then
    echo "skip missing: $CKPT"
    continue
  fi
  J=$(sbatch --parsable --export=STEP=${STEP} ffhq128_steps_ablation_eval.sh)
  echo "submitted FID eval step=${STEP} -> job $J"
done

squeue -u "$USER" | head
