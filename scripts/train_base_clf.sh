#!/usr/bin/env bash

DATASET="fashionmnist"
BASE_CLF="small_convnet_mnist"
BASE_DIR_NAME=${DATASET}"_smallerconvnet"
# BASE_EPOCH=100
RANDOM_SEED=777

OUTPUT_DIR=./output/${DATASET}/${BASE_DIR_NAME}/${RANDOM_SEED}


python ./train.py \
    --dataset ${DATASET} \
    --classifier_name ${BASE_CLF} \
    --seed ${RANDOM_SEED} \
    --gpu_ids "2, 3" \
    --batch_size 256 \
    --output_dir ${OUTPUT_DIR} \
    --num_workers 1 \
    --epochs 100 \
    --lr 3e-4 \
    --lr_scheduler None \
    --step_size 2 \
    --gamma 0.5 \
    --max_grad_clip 0 \
    --max_grad_norm 0 \
    --tensorboard

