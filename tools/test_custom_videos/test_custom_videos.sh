#!/bin/bash

CONFIG="configs/univs_inf_custom_videos/univs_swinb_vps_c1+univs_entity.yaml"
CHECKPOINT="pretrained/univs_v2_cvpr/univs_swinb_stage3_f7_wosquare_ema.pth"
OUTPUT_DIR="datasets/custom_videos/inference"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Enable visualization (optional)
# You can uncomment this if you want to visualize the results
# sed -i 's/self.visualize_results_enable = False/self.visualize_results_enable = True/g' univs/inference/inference_video_entity.py

# Run inference
python -m univs.engine.inference \
  --config-file ${CONFIG} \
  --num-gpus 1 \
  MODEL.WEIGHTS ${CHECKPOINT} \
  MODEL.UniVS.TEST.TASK "vis" \
  MODEL.UniVS.TEST.CUSTOM_VIDEOS_ENABLE True \
  OUTPUT_DIR ${OUTPUT_DIR}