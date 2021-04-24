#!/usr/bin/env bash

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
# ALGORITHM="RDA_V2"
# PROJ_NAME="refine_bing"
# SOURCE="Bing"
# TARGET="Bing"
# NOISY_TYPE="feature_uniform" #uniform, pair, none,feature_uniform
# NOISY_RATE="0.1"
# DEL_RATE="0"
# DATASET="Bing-Caltech"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/NoiseRemover.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     --del_rate ${DEL_RATE} \
# #    >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
# ALGORITHM="RDA_V2"
# PROJ_NAME="refine_bing"
# SOURCE="Bing"
# TARGET="Bing"
# NOISY_TYPE="feature_uniform" #uniform, pair, none,feature_uniform
# NOISY_RATE="0.2"
# DEL_RATE="0"
# DATASET="Bing-Caltech"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/NoiseRemover.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     --del_rate ${DEL_RATE} \
# #    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=0

PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
ALGORITHM="RDA_V2"
PROJ_NAME="refine_bing"
SOURCE="Bing"
TARGET="Bing"
NOISY_TYPE="feature_uniform" #uniform, pair, none,feature_uniform
NOISY_RATE="0.3"
DEL_RATE="0"
DATASET="Bing-Caltech"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

python trainer/NoiseRemover.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    --del_rate ${DEL_RATE} \
#    >> ${LOG_FILE}  2>&1


export CUDA_VISIBLE_DEVICES=0

PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
ALGORITHM="RDA_V2"
PROJ_NAME="refine_bing"
SOURCE="Bing"
TARGET="Bing"
NOISY_TYPE="feature_uniform" #uniform, pair, none,feature_uniform
NOISY_RATE="0.4"
DEL_RATE="0"
DATASET="Bing-Caltech"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

python trainer/NoiseRemover.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    --del_rate ${DEL_RATE} \
#    >> ${LOG_FILE}  2>&1