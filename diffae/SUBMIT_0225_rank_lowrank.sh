#!/bin/bash
# Submit LoRA-style low-rank projector ablation: ranks 5 / 10 / 512, save at 20k & 50k.
# Usage:
#   bash SUBMIT_0225_rank_lowrank.sh
#   RANKS="5 10" bash SUBMIT_0225_rank_lowrank.sh
set -e
cd /nfs/data_chaos/czhang/HouseholderGAN/diffae
RANKS=${RANKS:-"5 10 512"}

echo "Low-rank (LoRA) projector ablation; save steps: 20000 50000 (+ step=0)"
echo "Need ~12G+ free quota (3 ranks x ~2 ckpts x ~2G)."

for RANK in $RANKS; do
  J=$(sbatch --parsable --export=RANK=${RANK} ffhq128_rank_lowrank_train.sh)
  echo "submitted rank=${RANK} lowrank train -> job $J"
  J2=$(sbatch --parsable --dependency=afterany:$J --export=RANK=${RANK} ffhq128_rank_lowrank_train.sh)
  echo "submitted rank=${RANK} resume backup -> job $J2"
done

squeue -u "$USER" | head
