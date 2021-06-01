#!/usr/bin/env bash

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
# ALGORITHM="sample_selection"
# SOURCE="source"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none,feature_uniform
# NOISY_RATE="0.6"
# DATASET="COVID-19"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${SOURCE}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${SOURCE}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"
# #echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
# python trainer/sample_selection.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     --noisy_type ${NOISY_TYPE} \
#     #>> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
# ALGORITHM="RDA_V2"
# PROJ_NAME="refine_covid"
# SOURCE="source"
# TARGET="source"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none,feature_uniform
# NOISY_RATE="0.6"
# DEL_RATE="0"
# DATASET="COVID-19"

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

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
# ALGORITHM="RDA_V1"
# PROJ_NAME="S2T"
# SOURCE="source"
# TARGET="target"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DEL_RATE="0.2"
# DATASET="COVID-19"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/RDA_train_without_unlabeled_data.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     --del_rate ${DEL_RATE} \
#     >> ${LOG_FILE}  2>&1


PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
ALGORITHM="RDA_V2"
PROJ_NAME="S2T"
SOURCE="source"
TARGET="target"
NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
NOISY_RATE="0.6"
DEL_RATE="0.2"
DATASET="COVID-19"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

python trainer/RDA_train.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --noisy_rate ${NOISY_RATE} \
    --del_rate ${DEL_RATE} \
    >> ${LOG_FILE}  2>&1


# for i in 0.2 0.4 0.8

# do
#     export CUDA_VISIBLE_DEVICES=0

#     PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
#     ALGORITHM="sample_selection"
#     SOURCE="amazon"
#     NOISY_TYPE="ood_feature_uniform" #uniform, pair, none,feature_uniform
#     NOISY_RATE=$i
#     DATASET="Office-31"

#     LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${SOURCE}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
#     STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${SOURCE}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"
#     #echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
#     python trainer/sample_selection.py \
#         --config ${PROJ_ROOT}/config/dann.yml \
#         --dataset ${DATASET} \
#         --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#         --stats_file ${STATS_FILE} \
#         --noisy_rate ${NOISY_RATE} \
#         --noisy_type ${NOISY_TYPE} \
#         #>> ${LOG_FILE}  2>&1

#     PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
#     ALGORITHM="RDA_V2"
#     PROJ_NAME="refine_amazon"
#     SOURCE="amazon"
#     TARGET="amazon"
#     NOISY_TYPE="ood_feature_uniform" #uniform, pair, none,feature_uniform
#     NOISY_RATE=$i
#     DEL_RATE="0"
#     DATASET="Office-31"

#     LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
#     STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

#     python trainer/NoiseRemover.py \
#         --config ${PROJ_ROOT}/config/dann.yml \
#         --dataset ${DATASET} \
#         --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#         --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#         --stats_file ${STATS_FILE} \
#         --noisy_rate ${NOISY_RATE} \
#         --del_rate ${DEL_RATE} \
#     #    >> ${LOG_FILE}  2>&1   
# done

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
# ALGORITHM="RDA_V2"
# PROJ_NAME="A2W"
# SOURCE="amazon"
# TARGET="webcam"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.2"
# DEL_RATE="0."
# DATASET="Office-31"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/RDA_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     --del_rate ${DEL_RATE} \
#     >> ${LOG_FILE}  2>&1

# ROJ_ROOT="/home/ubuntu/nas/projects/RDA"
# ALGORITHM="RDA_V2"
# PROJ_NAME="A2W"
# SOURCE="amazon"
# TARGET="webcam"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.4"
# DEL_RATE="0.1"
# DATASET="Office-31"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/RDA_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     --del_rate ${DEL_RATE} \
#     >> ${LOG_FILE}  2>&1

# ROJ_ROOT="/home/ubuntu/nas/projects/RDA"
# ALGORITHM="RDA_V2"
# PROJ_NAME="A2W"
# SOURCE="amazon"
# TARGET="webcam"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.8"
# DEL_RATE="0.3"
# DATASET="Office-31"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/RDA_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     --del_rate ${DEL_RATE} \
#     >> ${LOG_FILE}  2>&1
# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
# ALGORITHM="sample_selection"
# SOURCE="amazon"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none,feature_uniform
# NOISY_RATE="0.6"
# DATASET="Office-31"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${SOURCE}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${SOURCE}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"
# #echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
# python trainer/sample_selection.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     --noisy_type ${NOISY_TYPE} \
#     #>> ${LOG_FILE}  2>&1


# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
# ALGORITHM="sample_selection"
# SOURCE="webcam"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none,feature_uniform
# NOISY_RATE="0.6"
# DATASET="Office-31"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${SOURCE}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${SOURCE}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"
# #echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
# python trainer/sample_selection.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     --noisy_type ${NOISY_TYPE} \
#     #>> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
# ALGORITHM="sample_selection"
# SOURCE="dslr"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none,feature_uniform
# NOISY_RATE="0.6"
# DATASET="Office-31"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${SOURCE}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${SOURCE}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"
# #echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
# python trainer/sample_selection.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     --noisy_type ${NOISY_TYPE} \
#     #>> ${LOG_FILE}  2>&1