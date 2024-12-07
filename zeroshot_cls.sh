#!/bin/bash

# Path to dataset
DATASET=modelnet40
#DATASET=modelnet10
#DATASET=scanobjectnn
#DATASET=scanobjectnn_obj
#DATASET=scanobjectnn_obj_bg

TRAINER=PointCLIPV2_ZS
# Trainer configs: rn50, rn101, vit_b32 or vit_b16
CFG=vit_b16

declare -a ALPHAS=(
  #"0.05"
  #"0.08"
  "0.1"
  "0.2"
  "0.3"
  "0.4"
  "0.5"
  "0.6"
  "0.7"
  "0.8"
  "0.9"
                  )

export CUDA_VISIBLE_DEVICES=0

for alpha in "${ALPHAS[@]}"
do
  echo "alpha: $alpha"
  python main.py \
  --trainer ${TRAINER} \
  --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
  --output-dir output/${TRAINER}/${CFG}/${DATASET} \
  --no-train \
  --zero-shot \
  --post-search \
  --alpha ${alpha} \
  --dataset-config-file configs/datasets/${DATASET}.yaml

done
