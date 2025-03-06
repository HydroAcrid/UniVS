#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2 python train_net.py \
    --num-gpus 3 \
    --config-file configs/univs_inf/vids/vis/univs_R50_hurricane.yaml \
    --eval-only \
    MODEL.WEIGHTS output/stage2/univs_r50_stage2.pth \
    OUTPUT_DIR output/results/hurricane/ \
    VIS_PERIOD 1
