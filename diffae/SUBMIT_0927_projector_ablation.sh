#!/bin/bash
# Submit the three corrected projector ablations (date 0927):
#   1) DiffAE + random Householder (SVD-ortho) rank=10
#   2) DiffAE + random Householder (SVD-ortho) rank=512
#   3) DiffAE + random LoRA A:512x10, B:10x512
# Saves at 20k / 50k (+ step=0).
set -e
cd /nfs/data_chaos/czhang/HouseholderGAN/diffae

echo "=== 0927 projector ablation ==="
echo "arms: householder@10, householder@512, lora@10 (all random projector init)"

submit_one () {
  local METHOD=$1 RANK=$2
  J=$(sbatch --parsable --export=METHOD=${METHOD},RANK=${RANK} ffhq128_projector_ablation_train.sh)
  echo "submitted ${METHOD} rank=${RANK} -> job $J"
  J2=$(sbatch --parsable --dependency=afterany:$J --export=METHOD=${METHOD},RANK=${RANK} ffhq128_projector_ablation_train.sh)
  echo "submitted ${METHOD} rank=${RANK} resume backup -> job $J2"
}

submit_one householder 10
submit_one householder 512
submit_one lora 10

squeue -u "$USER" | head
