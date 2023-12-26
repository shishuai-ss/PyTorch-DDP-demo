#!/bin/bash

# 只针对并行云训练，没有什么卵用
module load anaconda/2020.11
module load cuda/11.3

source activate ss_yolo

torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=12345 ddp_main.py --use_mix_precision --data-dir /data/public/cifar