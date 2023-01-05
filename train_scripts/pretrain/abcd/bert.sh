. .env

export MAX_STEPS=5000
export EVAL_STEPS=500

# Adjusts batch size to fit into GPU memory.
# Effective batch size = 
# PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * # GPUs.
export PER_DEVICE_TRAIN_BATCH_SIZE=512
export PER_DEVICE_EVAL_BATCH_SIZE=512
export GRADIENT_ACCUMULATION_STEPS=2

export LEARNING_RATE=5e-5
export WEIGHT_DECAY=1e-4
export METRIC_FOR_BEST_MODEL=dev_mlm_accuracy

DATASET=abcd
MODEL_NAME=bert
MODEL_BACKBONE=bert-base-uncased

export MODEL_PATH=$NFS_PARENT_DIR/pretrained_models/$MODEL_BACKBONE
export DATA_CONFIG_PATH=./config/data/${DATASET}.yaml

mkdir -p /tmp/$USER
export SCRATCH_DIR=`mktemp -d -p /tmp/$USER/`
trap 'rm -rf "$SCRATCH_DIR"' EXIT

echo '========================================================================='
echo 'Create data ...'
echo '========================================================================='
bash ./data_scripts/${DATASET}.sh --task_name pretrain_bert

echo '========================================================================='
echo 'Start training ...'
echo '========================================================================='
if [ -z $SLURM_ARRAY_TASK_ID ];
then
    export SEED=42
else
    export SEED=$SLURM_ARRAY_TASK_ID
fi

export EXP_NAME=pretrain/${MODEL_NAME}_init_from_${MODEL_BACKBONE}_job_name-$SLURM_JOB_NAME
export SUB_EXP_NAME=seed-$SEED
bash train_scripts/base_scripts/pretrain_bert.sh $DATASET