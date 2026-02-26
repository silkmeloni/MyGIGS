#!/bin/bash

# 循环 1 到 5
for i in {1..5}
do
    echo "========================================"
    echo "Starting training run #$i (Output: l_bicycle_color_$i)"
    echo "========================================"

    python train.py \
        -m "outputs/exam/360/l_bicycle_color_${i}" \
        -s ~/../mnt/e/CGBishe/dataset/360_v2/bicycle_r4/ \
        --iterations 40000 \
        --eval \
        --radius 0.8 \
        --bias 0.01 \
        --thick 0.05 \
        --delta 0.0625 \
        --step 16 \
        --start 64 \
        --metallic \
        --indirect \
        --degree 3 \
        --color_sabotage \
        --sabotage_rough_thresh 0.1 \
        --sabotage_patience 35000
        
    echo "Finished run #$i"
    echo ""
done

echo "All 5 runs completed!"