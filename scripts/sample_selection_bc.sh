#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

PROJ_ROOT="/home/zhongyi/projects/da/RDA"
ALGORITHM="sample_selection"
SOURCE="Bing"
NOISY_TYPE="feature_uniform" #uniform, pair, none
NOISY_RATE="0.2"
DATASET="Bing-Caltech"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${SOURCE}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${SOURCE}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"
echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python trainer/sample_selection.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    --noisy_type ${NOISY_TYPE} \
    >> ${LOG_FILE}  2>&1

