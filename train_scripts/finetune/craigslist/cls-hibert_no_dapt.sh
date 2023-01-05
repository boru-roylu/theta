. .env

export NUM_TRAIN_EPOCH=4

export PER_DEVICE_TRAIN_BATCH_SIZE=16
export PER_DEVICE_EVAL_BATCH_SIZE=16
export LEARNING_RATE=1e-5
export WEIGHT_DECAY=1e-4
export METRIC_FOR_BEST_MODEL=dev_accuracy

DATASET=craigslist
MODEL_NAME=theta-cls-hibert
MODEL_BACKBONE=bert-base-uncased

export MODEL_PATH=$NFS_PARENT_DIR/pretrained_models/$MODEL_BACKBONE
export DATA_CONFIG_PATH=./config/data/${DATASET}.yaml
export HIBERT_CONFIG_PATH=./config/model/${MODEL_NAME}/cls-hibert-absolute-pos-config-1layer-2head.json

mkdir -p /tmp/$USER
export SCRATCH_DIR=`mktemp -d -p /tmp/$USER/`
trap 'rm -rf "$SCRATCH_DIR"' EXIT

echo '==============================================================================='
echo 'Create data ...'
echo '==============================================================================='
bash ./data_scripts/${DATASET}.sh --task_name finetune

echo '==============================================================================='
echo 'Start training ...'
echo '==============================================================================='
if [ -z $SLURM_ARRAY_TASK_ID ];
then
    export SEED=21
else
    export SEED=$SLURM_ARRAY_TASK_ID
fi
POS_TYPE=absolute
export SEED=$SLURM_ARRAY_TASK_ID
export EXP_NAME=finetune/$MODEL_NAME-$POS_TYPE-pos_init_from_${MODEL_BACKBONE}_job_name-$SLURM_JOB_NAME
export SUB_EXP_NAME=seed-$SEED
bash train_scripts/base_scripts/finetune.sh $DATASET $MODEL_NAME $POS_TYPE $@