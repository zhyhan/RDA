#!/usr/bin/env bash
# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
# ALGORITHM="RDALTH"
# PROJ_NAME="A2W"
# SOURCE="amazon"
# TARGET="webcam"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DEL_RATE="0.2"
# DATASET="Office-31"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"

# python trainer/RDALTH_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#      >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
# ALGORITHM="RDALTH"
# PROJ_NAME="W2A"
# SOURCE="webcam"
# TARGET="amazon"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DEL_RATE="0.2"
# DATASET="Office-31"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"

# python trainer/RDALTH_train.py \
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
# ALGORITHM="RDALTH"
# PROJ_NAME="A2D"
# SOURCE="amazon"
# TARGET="dslr"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DEL_RATE="0.2"
# DATASET="Office-31"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"

# python trainer/RDALTH_train.py \
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
# ALGORITHM="RDALTH"
# PROJ_NAME="D2A"
# SOURCE="dslr"
# TARGET="amazon"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DEL_RATE="0.2"
# DATASET="Office-31"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"

# python trainer/RDALTH_train.py \
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
# ALGORITHM="RDALTH"
# PROJ_NAME="W2D"
# SOURCE="webcam"
# TARGET="dslr"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DEL_RATE="0.2"
# DATASET="Office-31"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"

# python trainer/RDALTH_train.py \
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
# ALGORITHM="RDALTH"
# PROJ_NAME="D2W"
# SOURCE="dslr"
# TARGET="webcam"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DEL_RATE="0.2"
# DATASET="Office-31"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"

# python trainer/RDALTH_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     --del_rate ${DEL_RATE} \
#     >> ${LOG_FILE}  2>&1




#for i in "ood_uniform" "feature_uniform" "uniform" "feature" "ood"
for i in "uniform" "feature" "ood"

do
    export CUDA_VISIBLE_DEVICES=0
    PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
    ALGORITHM="RDALTH"
    PROJ_NAME="A2W"
    SOURCE="amazon"
    TARGET="webcam"
    NOISY_TYPE=$i #uniform, pair, none
    NOISY_RATE="0.4"
    DEL_RATE="0.2"
    DATASET="Office-31"

    LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
    STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"

    python trainer/RDALTH_train.py \
        --config ${PROJ_ROOT}/config/dann.yml \
        --dataset ${DATASET} \
        --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
        --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
        --stats_file ${STATS_FILE} \
        --noisy_rate ${NOISY_RATE} \
        --del_rate ${DEL_RATE} \
        >> ${LOG_FILE}  2>&1

    export CUDA_VISIBLE_DEVICES=0

    PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
    ALGORITHM="RDALTH"
    PROJ_NAME="W2A"
    SOURCE="webcam"
    TARGET="amazon"
    NOISY_TYPE=$i #uniform, pair, none
    NOISY_RATE="0.4"
    DEL_RATE="0.2"
    #
    DATASET="Office-31"

    LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
    STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"

    python trainer/RDALTH_train.py \
        --config ${PROJ_ROOT}/config/dann.yml \
        --dataset ${DATASET} \
        --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
        --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
        --stats_file ${STATS_FILE} \
        --noisy_rate ${NOISY_RATE} \
        --del_rate ${DEL_RATE} \
        >> ${LOG_FILE}  2>&1

    export CUDA_VISIBLE_DEVICES=0

    PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
    ALGORITHM="RDALTH"
    PROJ_NAME="A2D"
    SOURCE="amazon"
    TARGET="dslr"
    NOISY_TYPE=$i #uniform, pair, none
    NOISY_RATE="0.4"
    DEL_RATE="0.2"
    #
    DATASET="Office-31"

    LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
    STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"

    python trainer/RDALTH_train.py \
        --config ${PROJ_ROOT}/config/dann.yml \
        --dataset ${DATASET} \
        --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
        --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
        --stats_file ${STATS_FILE} \
        --noisy_rate ${NOISY_RATE} \
        --del_rate ${DEL_RATE} \
        >> ${LOG_FILE}  2>&1

    export CUDA_VISIBLE_DEVICES=0

    PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
    ALGORITHM="RDALTH"
    PROJ_NAME="D2A"
    SOURCE="dslr"
    TARGET="amazon"
    NOISY_TYPE=$i #uniform, pair, none
    NOISY_RATE="0.4"
    DEL_RATE="0.2"
    #
    DATASET="Office-31"

    LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
    STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"

    python trainer/RDALTH_train.py \
        --config ${PROJ_ROOT}/config/dann.yml \
        --dataset ${DATASET} \
        --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
        --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
        --stats_file ${STATS_FILE} \
        --noisy_rate ${NOISY_RATE} \
        --del_rate ${DEL_RATE} \
        >> ${LOG_FILE}  2>&1

    export CUDA_VISIBLE_DEVICES=0

    PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
    ALGORITHM="RDALTH"
    PROJ_NAME="W2D"
    SOURCE="webcam"
    TARGET="dslr"
    NOISY_TYPE=$i #uniform, pair, none
    NOISY_RATE="0.4"
    DEL_RATE="0.2"
    #
    DATASET="Office-31"

    LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
    STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"

    python trainer/RDALTH_train.py \
        --config ${PROJ_ROOT}/config/dann.yml \
        --dataset ${DATASET} \
        --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
        --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
        --stats_file ${STATS_FILE} \
        --noisy_rate ${NOISY_RATE} \
        --del_rate ${DEL_RATE} \
        >> ${LOG_FILE}  2>&1

    export CUDA_VISIBLE_DEVICES=0

    PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
    ALGORITHM="RDALTH"
    PROJ_NAME="D2W"
    SOURCE="dslr"
    TARGET="webcam"
    NOISY_TYPE=$i #uniform, pair, none
    NOISY_RATE="0.4"
    DEL_RATE="0.2"
    #
    DATASET="Office-31"

    LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
    STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"

    python trainer/RDALTH_train.py \
        --config ${PROJ_ROOT}/config/dann.yml \
        --dataset ${DATASET} \
        --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
        --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
        --stats_file ${STATS_FILE} \
        --noisy_rate ${NOISY_RATE} \
        --del_rate ${DEL_RATE} \
        >> ${LOG_FILE}  2>&1

done

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
# ALGORITHM="RDALTH"
# PROJ_NAME="Ar2Cl"
# SOURCE="Art"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DEL_RATE="0.2"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"

# python trainer/RDALTH_train.py \
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
# ALGORITHM="RDALTH"
# PROJ_NAME="Ar2Pr"
# SOURCE="Art"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DEL_RATE="0.2"
# DATASET="Office-home"


# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"


# python trainer/RDALTH_train.py \
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
# ALGORITHM="RDALTH"
# PROJ_NAME="Ar2Rw"
# SOURCE="Art"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DEL_RATE="0.2"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"


# python trainer/RDALTH_train.py \
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
# ALGORITHM="RDALTH"
# PROJ_NAME="Cl2Ar"
# SOURCE="Clipart"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DEL_RATE="0.2"

# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"


# python trainer/RDALTH_train.py \
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
# ALGORITHM="RDALTH"
# PROJ_NAME="Cl2Pr"
# SOURCE="Clipart"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DEL_RATE="0.2"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"


# python trainer/RDALTH_train.py \
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
# ALGORITHM="RDALTH"
# PROJ_NAME="Cl2Rw"
# SOURCE="Clipart"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DEL_RATE="0.2"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"


# python trainer/RDALTH_train.py \
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
# ALGORITHM="RDALTH"
# PROJ_NAME="Pr2Ar"
# SOURCE="Product"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DEL_RATE="0.2"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"


# python trainer/RDALTH_train.py \
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
# ALGORITHM="RDALTH"
# PROJ_NAME="Pr2Cl"
# SOURCE="Product"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DEL_RATE="0.2"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"


# python trainer/RDALTH_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --del_rate ${DEL_RATE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
# ALGORITHM="RDALTH"
# PROJ_NAME="Pr2Rw"
# SOURCE="Product"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DEL_RATE="0.2"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"


# python trainer/RDALTH_train.py \
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
# ALGORITHM="RDALTH"
# PROJ_NAME="Rw2Ar"
# SOURCE="Real_world"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DEL_RATE="0.2"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"


# python trainer/RDALTH_train.py \
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
# ALGORITHM="RDALTH"
# PROJ_NAME="Rw2Cl"
# SOURCE="Real_world"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DEL_RATE="0.2"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"


# python trainer/RDALTH_train.py \
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
# ALGORITHM="RDALTH"
# PROJ_NAME="Rw2Pr"
# SOURCE="Real_world"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DEL_RATE="0.2"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"


# python trainer/RDALTH_train.py \
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
# ALGORITHM="RDALTH"
# PROJ_NAME="Ar2Cl"
# SOURCE="Art"
# TARGET="Clipart"
# NOISY_TYPE="feature_uniform" #uniform, pair, none
# NOISY_RATE="0.4"
# DEL_RATE="0.2"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"

# python trainer/RDALTH_train.py \
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
# ALGORITHM="RDALTH"
# PROJ_NAME="Ar2Pr"
# SOURCE="Art"
# TARGET="Product"
# NOISY_TYPE="feature_uniform" #uniform, pair, none
# NOISY_RATE="0.4"
# DEL_RATE="0.2"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"


# python trainer/RDALTH_train.py \
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
# ALGORITHM="RDALTH"
# PROJ_NAME="Ar2Rw"
# SOURCE="Art"
# TARGET="Real_world"
# NOISY_TYPE="feature_uniform" #uniform, pair, none
# NOISY_RATE="0.4"
# DEL_RATE="0.2"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"


# python trainer/RDALTH_train.py \
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
# ALGORITHM="RDALTH"
# PROJ_NAME="Cl2Ar"
# SOURCE="Clipart"
# TARGET="Art"
# NOISY_TYPE="feature_uniform" #uniform, pair, none
# NOISY_RATE="0.4"
# DEL_RATE="0.2"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"


# python trainer/RDALTH_train.py \
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
# ALGORITHM="RDALTH"
# PROJ_NAME="Cl2Pr"
# SOURCE="Clipart"
# TARGET="Product"
# NOISY_TYPE="feature_uniform" #uniform, pair, none
# NOISY_RATE="0.4"
# DEL_RATE="0.2"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"


# python trainer/RDALTH_train.py \
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
# ALGORITHM="RDALTH"
# PROJ_NAME="Cl2Rw"
# SOURCE="Clipart"
# TARGET="Real_world"
# NOISY_TYPE="feature_uniform" #uniform, pair, none
# NOISY_RATE="0.4"
# DEL_RATE="0.2"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"


# python trainer/RDALTH_train.py \
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
# ALGORITHM="RDALTH"
# PROJ_NAME="Pr2Ar"
# SOURCE="Product"
# TARGET="Art"
# NOISY_TYPE="feature_uniform" #uniform, pair, none
# NOISY_RATE="0.4"
# DEL_RATE="0.2"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"


# python trainer/RDALTH_train.py \
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
# ALGORITHM="RDALTH"
# PROJ_NAME="Pr2Cl"
# SOURCE="Product"
# TARGET="Clipart"
# NOISY_TYPE="feature_uniform" #uniform, pair, none
# NOISY_RATE="0.4"
# DEL_RATE="0.2"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"


# python trainer/RDALTH_train.py \
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
# ALGORITHM="RDALTH"
# PROJ_NAME="Pr2Rw"
# SOURCE="Product"
# TARGET="Real_world"
# NOISY_TYPE="feature_uniform" #uniform, pair, none
# NOISY_RATE="0.4"
# DEL_RATE="0.2"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"


# python trainer/RDALTH_train.py \
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
# ALGORITHM="RDALTH"
# PROJ_NAME="Rw2Ar"
# SOURCE="Real_world"
# TARGET="Art"
# NOISY_TYPE="feature_uniform" #uniform, pair, none
# NOISY_RATE="0.4"
# DEL_RATE="0.2"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"


# python trainer/RDALTH_train.py \
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
# ALGORITHM="RDALTH"
# PROJ_NAME="Rw2Cl"
# SOURCE="Real_world"
# TARGET="Clipart"
# NOISY_TYPE="feature_uniform" #uniform, pair, none
# NOISY_RATE="0.4"
# DEL_RATE="0.2"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"

# python trainer/RDALTH_train.py \
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
# ALGORITHM="RDALTH"
# PROJ_NAME="Rw2Pr"
# SOURCE="Real_world"
# TARGET="Product"
# NOISY_TYPE="feature_uniform" #uniform, pair, none
# NOISY_RATE="0.4"
# DEL_RATE="0.2"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d'`.pkl"

# python trainer/RDALTH_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     --del_rate ${DEL_RATE} \
#     >> ${LOG_FILE}  2>&1

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"
# ALGORITHM="RDALTH"
# PROJ_NAME="B2C"
# SOURCE="Bing"
# TARGET="Caltech"
# NOISY_TYPE="feature_uniform" #uniform, pair, none
# NOISY_RATE="0.4"
# DEL_RATE="0.2"
# DATASET="Bing-Caltech"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# echo "GPU: $CUDA_VISIBLE_DEVICES" > ${LOG_FILE}
# python trainer/RDALTH_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     --del_rate ${DEL_RATE} \
#     >> ${LOG_FILE}  2>&1