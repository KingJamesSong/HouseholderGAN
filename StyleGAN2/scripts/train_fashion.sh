#!/bin/bash
CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch --nproc_per_node=2 --master_port=9097 train.py --batch 16 ./data/fashion_lmdb --ckpt ./checkpoint_fashion/190000.pt