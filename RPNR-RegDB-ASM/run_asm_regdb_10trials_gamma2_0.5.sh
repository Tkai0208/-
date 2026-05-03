#!/bin/bash
# ASM Gamma 2.0/0.5 - 10 Trials Main Experiment

echo "=========================================="
echo "ASM Training - 10 Trials"
echo "Gamma: gamma_v=2.0, gamma_a=0.5 (Best)"
echo "=========================================="

for trial in 6 7 8 9 10
do
    echo ""
    echo "=========================================="
    echo "Starting Trial $trial"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=0 python train_regdb.py \
        -mb CMhcl \
        -b 128 \
        -a agw \
        -d regdb_rgb \
        --iters 100 \
        --momentum 0.1 \
        --eps 0.6 \
        --num-instances 16 \
        --trial $trial \
        --data-dir "../RegDB/"
    
    CUDA_VISIBLE_DEVICES=0 python test_regdb.py \
        -b 128 \
        -a agw \
        -d regdb_rgb \
        --trial $trial \
        --resume "logs/regdb_gamma_s2/${trial}/model_best.pth.tar" \
        --data-dir "../RegDB/"
    
    echo "Trial $trial completed!"
done

echo ""
echo "All remaining trials 6-10 (gamma 2.0/0.5) completed!"
