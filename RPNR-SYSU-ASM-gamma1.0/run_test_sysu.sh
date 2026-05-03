#!/bin/bash
# Test SYSU-MM01 with Augmented Matching model

CUDA_VISIBLE_DEVICES=0,1 \
python test_sysu.py \
    -b 64 \
    -a agw \
    -d sysu_all \
    --resume logs/sysu_asm_s2/model_best.pth.tar \
    --data-dir "/root/work/SYSU-MM01/"
