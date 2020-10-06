#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

PROJ_ROOT="/home/hanzhongyi/projects/da/RDA"
ALGORITHM="MDD"
PROJ_NAME="B2C"
SOURCE="Bing"
TARGET="Caltech"
NOISY_TYPE="unknown" #uniform, pair, none
NOISY_RATE="0"
DATASET="Bing-Caltech"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python trainer/MDD_train.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    >> ${LOG_FILE}  2>&1

