#!/bin/bash
for trial in 1 2 3 4 5 6 7 8 9 10
do
    echo "=========================================="
    echo "Starting Trial $trial"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=0 python train_regdb.py \
        -mb CMhcl -b 128 -a agw -d regdb_rgb \
        --iters 100 --momentum 0.1 --eps 0.6 --num-instances 16 \
        --trial $trial \
        --data-dir "../RegDB/"
    
    CUDA_VISIBLE_DEVICES=0 python test_regdb.py \
        -b 128 -a agw -d regdb_rgb \
        --trial $trial \
        --data-dir "../RegDB/"
    
    echo "Trial $trial completed!"
    echo ""
done

echo "All trials 1-10 completed!"

# 1 Trial
#CUDA_VISIBLE_DEVICES=0 \
#python train_regdb.py -mb CMhcl -b 128 -a agw -d regdb_rgb \
#--iters 100 --momentum 0.1 --eps 0.6 --num-instances 16 --trial 1 \
#--data-dir "../RegDB/"