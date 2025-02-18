#!/usr/bin/env bash

# Specify your config and model weights
CONFIG_FILE="configs/univs_r50_stage3.yaml"
WEIGHT_FILE="output/stage3/model_final.pth"

# Specify input and output directories
VIDEO_DIR="./datasets/custom_videos/raw"
OUTPUT_DIR="./datasets/custom_videos/inference"

# Run inference
python demo/demo_custom_videos.py \
    --config-file ${CONFIG_FILE} \
    --video-dir ${VIDEO_DIR} \
    --output ${OUTPUT_DIR} \
    --opts MODEL.WEIGHTS ${WEIGHT_FILE} \
    MODEL.MASK_FORMER.TEST.SEMANTIC_ON True \
    MODEL.MASK_FORMER.TEST.INSTANCE_ON True \
    MODEL.MASK_FORMER.TEST.PANOPTIC_ON True