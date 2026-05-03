for trial in 1 2 3 4 5 6 7 8 9 10
do
    echo "Starting Trial ${trial}"

    echo "  Training..."
    CUDA_VISIBLE_DEVICES=0,1 \
     python train_regdb.py -mb CMhybrid -b 128 -a agw -d regdb_rgb \
     --iters 100 --momentum 0.1 --eps 0.3 --num-instances 16 --trial ${trial} \
     --data-dir "/root/autodl-tmp/RegDB/"

    echo "  Evaluating..."
    CUDA_VISIBLE_DEVICES=0,1 \
     python test_regdb.py -b 128 -a agw -d regdb_rgb \
     --iters 100 --momentum 0.1 --eps 0.6 --num-instances 16 \
     --data-dir "/root/autodl-tmp/RegDB/" --trial ${trial}
done

echo "All 10 trials completed"
