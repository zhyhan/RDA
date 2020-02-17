#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

PROJ_ROOT="/home/zhongyi/projects/da/RDA"
ALGORITHM="MDD"
PROJ_NAME="Ar2Cl"
SOURCE="Art"
TARGET="Clipart"
NOISY_TYPE="feature_uniform" #uniform, pair, none
NOISY_RATE="0.4"
DATASET="Office-Home"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python trainer/MDD_train.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=3

PROJ_ROOT="/home/zhongyi/projects/da/RDA"
ALGORITHM="MDD"
PROJ_NAME="Ar2Pr"
SOURCE="Art"
TARGET="Product"
NOISY_TYPE="feature_uniform" #uniform, pair, none
NOISY_RATE="0.4"
DATASET="Office-Home"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python trainer/MDD_train.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=3

PROJ_ROOT="/home/zhongyi/projects/da/RDA"
ALGORITHM="MDD"
PROJ_NAME="Ar2Rw"
SOURCE="Art"
TARGET="Real_world"
NOISY_TYPE="feature_uniform" #uniform, pair, none
NOISY_RATE="0.4"
DATASET="Office-Home"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python trainer/MDD_train.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=3

PROJ_ROOT="/home/zhongyi/projects/da/RDA"
ALGORITHM="MDD"
PROJ_NAME="Cl2Ar"
SOURCE="Clipart"
TARGET="Art"
NOISY_TYPE="feature_uniform" #uniform, pair, none
NOISY_RATE="0.4"
DATASET="Office-Home"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python trainer/MDD_train.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=3

PROJ_ROOT="/home/zhongyi/projects/da/RDA"
ALGORITHM="MDD"
PROJ_NAME="Cl2Pr"
SOURCE="Clipart"
TARGET="Product"
NOISY_TYPE="feature_uniform" #uniform, pair, none
NOISY_RATE="0.4"
DATASET="Office-Home"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python trainer/MDD_train.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=3

PROJ_ROOT="/home/zhongyi/projects/da/RDA"
ALGORITHM="MDD"
PROJ_NAME="Cl2Rw"
SOURCE="Clipart"
TARGET="Real_world"
NOISY_TYPE="feature_uniform" #uniform, pair, none
NOISY_RATE="0.4"
DATASET="Office-Home"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python trainer/MDD_train.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=3

PROJ_ROOT="/home/zhongyi/projects/da/RDA"
ALGORITHM="MDD"
PROJ_NAME="Pr2Ar"
SOURCE="Product"
TARGET="Art"
NOISY_TYPE="feature_uniform" #uniform, pair, none
NOISY_RATE="0.4"
DATASET="Office-Home"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python trainer/MDD_train.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=3

PROJ_ROOT="/home/zhongyi/projects/da/RDA"
ALGORITHM="MDD"
PROJ_NAME="Pr2Cl"
SOURCE="Product"
TARGET="Clipart"
NOISY_TYPE="feature_uniform" #uniform, pair, none
NOISY_RATE="0.4"
DATASET="Office-Home"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python trainer/MDD_train.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=3

PROJ_ROOT="/home/zhongyi/projects/da/RDA"
ALGORITHM="MDD"
PROJ_NAME="Pr2Rw"
SOURCE="Product"
TARGET="Real_world"
NOISY_TYPE="feature_uniform" #uniform, pair, none
NOISY_RATE="0.4"
DATASET="Office-Home"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python trainer/MDD_train.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=3

PROJ_ROOT="/home/zhongyi/projects/da/RDA"
ALGORITHM="MDD"
PROJ_NAME="Rw2Ar"
SOURCE="Real_world"
TARGET="Art"
NOISY_TYPE="feature_uniform" #uniform, pair, none
NOISY_RATE="0.4"
DATASET="Office-Home"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python trainer/MDD_train.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=3

PROJ_ROOT="/home/zhongyi/projects/da/RDA"
ALGORITHM="MDD"
PROJ_NAME="Rw2Cl"
SOURCE="Real_world"
TARGET="Clipart"
NOISY_TYPE="feature_uniform" #uniform, pair, none
NOISY_RATE="0.4"
DATASET="Office-Home"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python trainer/MDD_train.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=3

PROJ_ROOT="/home/zhongyi/projects/da/RDA"
ALGORITHM="MDD"
PROJ_NAME="Rw2Pr"
SOURCE="Real_world"
TARGET="Product"
NOISY_TYPE="feature_uniform" #uniform, pair, none
NOISY_RATE="0.4"
DATASET="Office-Home"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
python trainer/MDD_train.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    >> ${LOG_FILE}  2>&1


