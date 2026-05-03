#!/bin/bash
# Augmented Matching (ASM) for SYSU-MM01 - 50 epochs test
# 使用gamma_v=2.0, gamma_a=1.0 (比例2:1，SYSU配置)

CUDA_VISIBLE_DEVICES=0,1 \
python train_sysu.py \
    -mb CMhcl \
    -b 128 \
    -a agw \
    -d sysu_all \
    --epochs 50 \
    --num-instances 16 \
    --iters 200 \
    --momentum 0.1 \
    --eps 0.6 \
    --data-dir "/root/work/SYSU-MM01/"
