#!/bin/bash
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 train.py --batch 16 ./data/cars_lmdb