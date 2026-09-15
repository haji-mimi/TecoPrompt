#!/bin/bash

#cd ../..

# custom config
DATA=../data
TRAINER=TecoPrompt

DATASET=$1
CFG=rn50  # config file
SHOTS=$2  # number of shots (1, 2, 4, 8, 16)
RATE=$3
TYPE=$4
CLASS=$5
K=${6:-8}  # temporal stability window (default: 8 epochs)

for SEED in 3 4 5
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/noise_${TYPE}_${RATE}/K${K}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.NOISE_RATE ${RATE} \
        DATASET.NOISE_TYPE ${TYPE} \
        DATASET.num_class ${CLASS} \
        WINDOW_K ${K}
    fi
done
