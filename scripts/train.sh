#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
ALGORITHM="RDA"
PROJ_NAME="A2D"
SOURCE="amazon"
TARGET="dslr"
NOISY_TYPE="uniform" #uniform, pair, none
NOISY_RATE="0.4"
DEL_RATE="0"
DATASET="Office-31"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

#echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python trainer/RDA_train.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    --del_rate ${DEL_RATE} \
#    >> ${LOG_FILE}  2>&1