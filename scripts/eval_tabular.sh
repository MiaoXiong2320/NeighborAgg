#!/usr/bin/env bash

cd ../

DATASET=LetterRecognition
BASE_CLD=LR

RANDOM_SEED=777
OUTPUT_DIR=output/ # specified by users


python ./eval_tabular_neighboragg.py \
    --dataset $DATASET \
    --classifier_name $BASE_CLF \
    --seed ${RANDOM_SEED} \
    --gpu_ids "0" \
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
