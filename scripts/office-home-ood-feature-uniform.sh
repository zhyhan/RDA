 #!/usr/bin/env bash

# #########ResNet##########
# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="ResNet"
# PROJ_NAME="Ar2Cl"
# SOURCE="Art"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/Resnet_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="ResNet"
# PROJ_NAME="Ar2Pr"
# SOURCE="Art"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/Resnet_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="ResNet"
# PROJ_NAME="Ar2Rw"
# SOURCE="Art"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/Resnet_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="ResNet"
# PROJ_NAME="Cl2Ar"
# SOURCE="Clipart"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/Resnet_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="ResNet"
# PROJ_NAME="Cl2Pr"
# SOURCE="Clipart"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/Resnet_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="ResNet"
# PROJ_NAME="Cl2Rw"
# SOURCE="Clipart"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/Resnet_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="ResNet"
# PROJ_NAME="Pr2Ar"
# SOURCE="Product"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/Resnet_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="ResNet"
# PROJ_NAME="Pr2Cl"
# SOURCE="Product"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/Resnet_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="ResNet"
# PROJ_NAME="Pr2Rw"
# SOURCE="Product"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/Resnet_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="ResNet"
# PROJ_NAME="Rw2Ar"
# SOURCE="Real_world"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"


# python trainer/Resnet_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="ResNet"
# PROJ_NAME="Rw2Cl"
# SOURCE="Real_world"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"


# python trainer/Resnet_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="ResNet"
# PROJ_NAME="Rw2Pr"
# SOURCE="Real_world"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/Resnet_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1
    
# ##########SPL#############
# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="SPL"
# PROJ_NAME="Ar2Cl"
# SOURCE="Art"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/SPL_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="SPL"
# PROJ_NAME="Ar2Pr"
# SOURCE="Art"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/SPL_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="SPL"
# PROJ_NAME="Ar2Rw"
# SOURCE="Art"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/SPL_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="SPL"
# PROJ_NAME="Cl2Ar"
# SOURCE="Clipart"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/SPL_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="SPL"
# PROJ_NAME="Cl2Pr"
# SOURCE="Clipart"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/SPL_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="SPL"
# PROJ_NAME="Cl2Rw"
# SOURCE="Clipart"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/SPL_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="SPL"
# PROJ_NAME="Pr2Ar"
# SOURCE="Product"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/SPL_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="SPL"
# PROJ_NAME="Pr2Cl"
# SOURCE="Product"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/SPL_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="SPL"
# PROJ_NAME="Pr2Rw"
# SOURCE="Product"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/SPL_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="SPL"
# PROJ_NAME="Rw2Ar"
# SOURCE="Real_world"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"


# python trainer/SPL_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="SPL"
# PROJ_NAME="Rw2Cl"
# SOURCE="Real_world"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"


# python trainer/SPL_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="SPL"
# PROJ_NAME="Rw2Pr"
# SOURCE="Real_world"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/SPL_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# #########MentorNet#############
# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="MENTOR"
# PROJ_NAME="Ar2Cl"
# SOURCE="Art"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/Mentor_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="MENTOR"
# PROJ_NAME="Ar2Pr"
# SOURCE="Art"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/Mentor_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="MENTOR"
# PROJ_NAME="Ar2Rw"
# SOURCE="Art"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/Mentor_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="MENTOR"
# PROJ_NAME="Cl2Ar"
# SOURCE="Clipart"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/Mentor_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="MENTOR"
# PROJ_NAME="Cl2Pr"
# SOURCE="Clipart"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/Mentor_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="MENTOR"
# PROJ_NAME="Cl2Rw"
# SOURCE="Clipart"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/Mentor_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="MENTOR"
# PROJ_NAME="Pr2Ar"
# SOURCE="Product"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/Mentor_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="MENTOR"
# PROJ_NAME="Pr2Cl"
# SOURCE="Product"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/Mentor_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="MENTOR"
# PROJ_NAME="Pr2Rw"
# SOURCE="Product"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/Mentor_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="MENTOR"
# PROJ_NAME="Rw2Ar"
# SOURCE="Real_world"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/Mentor_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="MENTOR"
# PROJ_NAME="Rw2Cl"
# SOURCE="Real_world"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"


# python trainer/Mentor_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="MENTOR"
# PROJ_NAME="Rw2Pr"
# SOURCE="Real_world"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/Mentor_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# ##########DAN#############
# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="DAN"
# PROJ_NAME="Ar2Cl"
# SOURCE="Art"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/DAN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="DAN"
# PROJ_NAME="Ar2Pr"
# SOURCE="Art"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/DAN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="DAN"
# PROJ_NAME="Ar2Rw"
# SOURCE="Art"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/DAN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="DAN"
# PROJ_NAME="Cl2Ar"
# SOURCE="Clipart"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/DAN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="DAN"
# PROJ_NAME="Cl2Pr"
# SOURCE="Clipart"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/DAN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="DAN"
# PROJ_NAME="Cl2Rw"
# SOURCE="Clipart"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/DAN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="DAN"
# PROJ_NAME="Pr2Ar"
# SOURCE="Product"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/DAN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="DAN"
# PROJ_NAME="Pr2Cl"
# SOURCE="Product"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/DAN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="DAN"
# PROJ_NAME="Pr2Rw"
# SOURCE="Product"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/DAN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="DAN"
# PROJ_NAME="Rw2Ar"
# SOURCE="Real_world"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/DAN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="DAN"
# PROJ_NAME="Rw2Cl"
# SOURCE="Real_world"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/DAN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="DAN"
# PROJ_NAME="Rw2Pr"
# SOURCE="Real_world"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/DAN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# ##########RTN#############
# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="RTN"
# PROJ_NAME="Ar2Cl"
# SOURCE="Art"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/RTN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="RTN"
# PROJ_NAME="Ar2Pr"
# SOURCE="Art"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/RTN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="RTN"
# PROJ_NAME="Ar2Rw"
# SOURCE="Art"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/RTN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="RTN"
# PROJ_NAME="Cl2Ar"
# SOURCE="Clipart"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/RTN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="RTN"
# PROJ_NAME="Cl2Pr"
# SOURCE="Clipart"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/RTN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="RTN"
# PROJ_NAME="Cl2Rw"
# SOURCE="Clipart"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/RTN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="RTN"
# PROJ_NAME="Pr2Ar"
# SOURCE="Product"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/RTN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="RTN"
# PROJ_NAME="Pr2Cl"
# SOURCE="Product"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/RTN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="RTN"
# PROJ_NAME="Pr2Rw"
# SOURCE="Product"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/RTN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="RTN"
# PROJ_NAME="Rw2Ar"
# SOURCE="Real_world"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/RTN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="RTN"
# PROJ_NAME="Rw2Cl"
# SOURCE="Real_world"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/RTN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="RTN"
# PROJ_NAME="Rw2Pr"
# SOURCE="Real_world"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/RTN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

##########DANN#############
# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="DANN"
# PROJ_NAME="Ar2Cl"
# SOURCE="Art"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/DANN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="DANN"
# PROJ_NAME="Ar2Pr"
# SOURCE="Art"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/DANN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="DANN"
# PROJ_NAME="Ar2Rw"
# SOURCE="Art"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/DANN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="DANN"
# PROJ_NAME="Cl2Ar"
# SOURCE="Clipart"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/DANN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="DANN"
# PROJ_NAME="Cl2Pr"
# SOURCE="Clipart"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/DANN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="DANN"
# PROJ_NAME="Cl2Rw"
# SOURCE="Clipart"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/DANN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="DANN"
# PROJ_NAME="Pr2Ar"
# SOURCE="Product"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/DANN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="DANN"
# PROJ_NAME="Pr2Cl"
# SOURCE="Product"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/DANN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="DANN"
# PROJ_NAME="Pr2Rw"
# SOURCE="Product"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/DANN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="DANN"
# PROJ_NAME="Rw2Ar"
# SOURCE="Real_world"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/DANN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="DANN"
# PROJ_NAME="Rw2Cl"
# SOURCE="Real_world"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/DANN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="DANN"
# PROJ_NAME="Rw2Pr"
# SOURCE="Real_world"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/DANN_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# ##########ADDA#############
# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="ADDA"
# PROJ_NAME="Ar2Cl"
# SOURCE="Art"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/ADDA_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="ADDA"
# PROJ_NAME="Ar2Pr"
# SOURCE="Art"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/ADDA_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="ADDA"
# PROJ_NAME="Ar2Rw"
# SOURCE="Art"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/ADDA_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="ADDA"
# PROJ_NAME="Cl2Ar"
# SOURCE="Clipart"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/ADDA_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="ADDA"
# PROJ_NAME="Cl2Pr"
# SOURCE="Clipart"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/ADDA_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="ADDA"
# PROJ_NAME="Cl2Rw"
# SOURCE="Clipart"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/ADDA_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="ADDA"
# PROJ_NAME="Pr2Ar"
# SOURCE="Product"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/ADDA_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="ADDA"
# PROJ_NAME="Pr2Cl"
# SOURCE="Product"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/ADDA_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="ADDA"
# PROJ_NAME="Pr2Rw"
# SOURCE="Product"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/ADDA_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="ADDA"
# PROJ_NAME="Rw2Ar"
# SOURCE="Real_world"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/ADDA_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="ADDA"
# PROJ_NAME="Rw2Cl"
# SOURCE="Real_world"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/ADDA_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="ADDA"
# PROJ_NAME="Rw2Pr"
# SOURCE="Real_world"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/ADDA_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

#########MDD#########
# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="MDD"
# PROJ_NAME="Ar2Cl"
# SOURCE="Art"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/MDD_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="MDD"
# PROJ_NAME="Ar2Pr"
# SOURCE="Art"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/MDD_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="MDD"
# PROJ_NAME="Ar2Rw"
# SOURCE="Art"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"


# python trainer/MDD_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="MDD"
# PROJ_NAME="Cl2Ar"
# SOURCE="Clipart"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/MDD_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="MDD"
# PROJ_NAME="Cl2Pr"
# SOURCE="Clipart"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/MDD_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="MDD"
# PROJ_NAME="Cl2Rw"
# SOURCE="Clipart"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/MDD_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="MDD"
# PROJ_NAME="Pr2Ar"
# SOURCE="Product"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/MDD_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="MDD"
# PROJ_NAME="Pr2Cl"
# SOURCE="Product"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/MDD_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="MDD"
# PROJ_NAME="Pr2Rw"
# SOURCE="Product"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/MDD_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="MDD"
# PROJ_NAME="Rw2Ar"
# SOURCE="Real_world"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"


# python trainer/MDD_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="MDD"
# PROJ_NAME="Rw2Cl"
# SOURCE="Real_world"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"


# python trainer/MDD_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="MDD"
# PROJ_NAME="Rw2Pr"
# SOURCE="Real_world"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"


# python trainer/MDD_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# ##########TCL#############
# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="TCL"
# PROJ_NAME="Ar2Cl"
# SOURCE="Art"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/TCL_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="TCL"
# PROJ_NAME="Ar2Pr"
# SOURCE="Art"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/TCL_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="TCL"
# PROJ_NAME="Ar2Rw"
# SOURCE="Art"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/TCL_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="TCL"
# PROJ_NAME="Cl2Ar"
# SOURCE="Clipart"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/TCL_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="TCL"
# PROJ_NAME="Cl2Pr"
# SOURCE="Clipart"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/TCL_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="TCL"
# PROJ_NAME="Cl2Rw"
# SOURCE="Clipart"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/TCL_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="TCL"
# PROJ_NAME="Pr2Ar"
# SOURCE="Product"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/TCL_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="TCL"
# PROJ_NAME="Pr2Cl"
# SOURCE="Product"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/TCL_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="TCL"
# PROJ_NAME="Pr2Rw"
# SOURCE="Product"
# TARGET="Real_world"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/TCL_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="TCL"
# PROJ_NAME="Rw2Ar"
# SOURCE="Real_world"
# TARGET="Art"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/TCL_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="TCL"
# PROJ_NAME="Rw2Cl"
# SOURCE="Real_world"
# TARGET="Clipart"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/TCL_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1

# export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

# ALGORITHM="TCL"
# PROJ_NAME="Rw2Pr"
# SOURCE="Real_world"
# TARGET="Product"
# NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
# NOISY_RATE="0.6"
# DATASET="Office-home"

# LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

# python trainer/TCL_train.py \
#     --config ${PROJ_ROOT}/config/dann.yml \
#     --dataset ${DATASET} \
#     --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
#     --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
#     --stats_file ${STATS_FILE} \
#     --noisy_rate ${NOISY_RATE} \
#     >> ${LOG_FILE}  2>&1



##########RDA_V1#############
export CUDA_VISIBLE_DEVICES=0

PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

ALGORITHM="RDA_V1"
PROJ_NAME="Ar2Cl"
SOURCE="Art"
TARGET="Clipart"
NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
NOISY_RATE="0.6"
DATASET="Office-home"
DEL_RATE="0.2"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

python trainer/RDA_train_without_unlabeled_data.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --del_rate ${DEL_RATE} \
    --noisy_rate ${NOISY_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=0

PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

ALGORITHM="RDA_V1"
PROJ_NAME="Ar2Pr"
SOURCE="Art"
TARGET="Product"
NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
NOISY_RATE="0.6"
DATASET="Office-home"
DEL_RATE="0.2"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

python trainer/RDA_train_without_unlabeled_data.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --del_rate ${DEL_RATE} \
    --noisy_rate ${NOISY_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=0

PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

ALGORITHM="RDA_V1"
PROJ_NAME="Ar2Rw"
SOURCE="Art"
TARGET="Real_world"
NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
NOISY_RATE="0.6"
DATASET="Office-home"
DEL_RATE="0.2"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

python trainer/RDA_train_without_unlabeled_data.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --del_rate ${DEL_RATE} \
    --noisy_rate ${NOISY_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=0

PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

ALGORITHM="RDA_V1"
PROJ_NAME="Cl2Ar"
SOURCE="Clipart"
TARGET="Art"
NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
NOISY_RATE="0.6"
DATASET="Office-home"
DEL_RATE="0.2"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

python trainer/RDA_train_without_unlabeled_data.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --del_rate ${DEL_RATE} \
    --noisy_rate ${NOISY_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=0

PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

ALGORITHM="RDA_V1"
PROJ_NAME="Cl2Pr"
SOURCE="Clipart"
TARGET="Product"
NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
NOISY_RATE="0.6"
DATASET="Office-home"
DEL_RATE="0.2"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

python trainer/RDA_train_without_unlabeled_data.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --del_rate ${DEL_RATE} \
    --noisy_rate ${NOISY_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=0

PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

ALGORITHM="RDA_V1"
PROJ_NAME="Cl2Rw"
SOURCE="Clipart"
TARGET="Real_world"
NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
NOISY_RATE="0.6"
DATASET="Office-home"
DEL_RATE="0.2"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

python trainer/RDA_train_without_unlabeled_data.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --del_rate ${DEL_RATE} \
    --noisy_rate ${NOISY_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=0

PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

ALGORITHM="RDA_V1"
PROJ_NAME="Pr2Ar"
SOURCE="Product"
TARGET="Art"
NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
NOISY_RATE="0.6"
DATASET="Office-home"
DEL_RATE="0.2"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

python trainer/RDA_train_without_unlabeled_data.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --del_rate ${DEL_RATE} \
    --noisy_rate ${NOISY_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=0

PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

ALGORITHM="RDA_V1"
PROJ_NAME="Pr2Cl"
SOURCE="Product"
TARGET="Clipart"
NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
NOISY_RATE="0.6"
DATASET="Office-home"
DEL_RATE="0.2"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

python trainer/RDA_train_without_unlabeled_data.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --del_rate ${DEL_RATE} \
    --noisy_rate ${NOISY_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=0

PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

ALGORITHM="RDA_V1"
PROJ_NAME="Pr2Rw"
SOURCE="Product"
TARGET="Real_world"
NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
NOISY_RATE="0.6"
DATASET="Office-home"
DEL_RATE="0.2"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

python trainer/RDA_train_without_unlabeled_data.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --del_rate ${DEL_RATE} \
    --noisy_rate ${NOISY_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=0

PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

ALGORITHM="RDA_V1"
PROJ_NAME="Rw2Ar"
SOURCE="Real_world"
TARGET="Art"
NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
NOISY_RATE="0.6"
DATASET="Office-home"
DEL_RATE="0.2"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

python trainer/RDA_train_without_unlabeled_data.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --del_rate ${DEL_RATE} \
    --noisy_rate ${NOISY_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=0

PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

ALGORITHM="RDA_V1"
PROJ_NAME="Rw2Cl"
SOURCE="Real_world"
TARGET="Clipart"
NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
NOISY_RATE="0.6"
DATASET="Office-home"
DEL_RATE="0.2"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

python trainer/RDA_train_without_unlabeled_data.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --del_rate ${DEL_RATE} \
    --noisy_rate ${NOISY_RATE} \
    >> ${LOG_FILE}  2>&1

export CUDA_VISIBLE_DEVICES=0

PROJ_ROOT="/home/ubuntu/nas/projects/RDA"

ALGORITHM="RDA_V1"
PROJ_NAME="Rw2Pr"
SOURCE="Real_world"
TARGET="Product"
NOISY_TYPE="ood_feature_uniform" #uniform, pair, none
NOISY_RATE="0.6"
DATASET="Office-home"
DEL_RATE="0.2"

LOG_FILE="${PROJ_ROOT}/log/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.log"
STATS_FILE="${PROJ_ROOT}/statistic/${ALGORITHM}-${PROJ_NAME}-${NOISY_TYPE}-noisy-${NOISY_RATE}-`date +'%Y-%m-%d-%H-%M-%S'`.pkl"

python trainer/RDA_train_without_unlabeled_data.py \
    --config ${PROJ_ROOT}/config/dann.yml \
    --dataset ${DATASET} \
    --src_address ${PROJ_ROOT}/data/${DATASET}/${SOURCE}_${NOISY_TYPE}_noisy_${NOISY_RATE}.txt \
    --tgt_address ${PROJ_ROOT}/data/${DATASET}/${TARGET}.txt \
    --stats_file ${STATS_FILE} \
    --del_rate ${DEL_RATE} \
    --noisy_rate ${NOISY_RATE} \
    >> ${LOG_FILE}  2>&1