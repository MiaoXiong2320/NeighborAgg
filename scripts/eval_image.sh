#!/usr/bin/env bash

DATASET="cifar10"

BASE_CLF="resnet50_cifar"
BASE_DIR_NAME=${DATASET}"_resnet50"
RANDOM_SEED=777

BASE_EPOCH=${EPOCH}
BASE_DIR=../classifier/code/simple_cls # specify base classifier's checkpoint


BASE_CKPT=$BASE_DIR/output/${DATASET}/${BASE_DIR_NAME}/${RANDOM_SEED}/ckpts/ckpt_e$BASE_EPOCH.pt
FEATURE_DIR=$BASE_DIR/output/${DATASET}/${BASE_DIR_NAME}/${RANDOM_SEED}/feats/e${BASE_EPOCH}/e${BASE_EPOCH}
OUTPUT_DIR="./output/${DATASET}/${BASE_DIR_NAME}/${RANDOM_SEED}/e${BASE_EPOCH}"


python ./eval_image_neighboragg.py \
    --dataset $DATASET \
    --feature_dir $FEATURE_DIR \
    --base_ckpt $BASE_CKPT \
    --classifier_name $BASE_CLF \
    --seed ${RANDOM_SEED} \
    --gpu_ids "3, 4, 5" \
    --batch_size 256 \
    --output_dir $OUTPUT_DIR \
    --num_workers 1 \
    --lr_scheduler "StepLR" \
    --step_size 2 \
    --gamma 0.5 \
    --max_grad_clip 0 \
    --max_grad_norm 0 \
    --tensorboard \
    --trust_nn "TrustNN_CNN" \
    --neigh_temp 1.0 \
    --K 10 \
    --loss_type "CE"