#!/bin/bash
# Re-run fair PIPL for rank 5 / 10 / 512 (fixed top-N_EIG directions).
# Override: N_EIG=5 RANKS="5 10 512" bash SUBMIT_ffhq128_rank_pipl_fair.sh
set -e
cd /nfs/data_chaos/czhang/HouseholderGAN/diffae
RANKS=${RANKS:-"5 10 512"}
N_EIG=${N_EIG:-5}

echo "Fair PIPL with n_eig=${N_EIG} for ranks: ${RANKS}"
for RANK in $RANKS; do
  J=$(sbatch --parsable --export=RANK=${RANK},N_EIG=${N_EIG} ffhq128_rank_pipl_fair.sh)
  echo "submitted fair PIPL rank=${RANK} -> job $J"
done
squeue -u "$USER" | head
