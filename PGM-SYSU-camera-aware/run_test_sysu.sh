#!/bin/bash
export OMP_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=0,1 python test_sysu.py -b 128 -a agw -d sysu_all --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 --data-dir "/root/work/SYSU-MM01/"
