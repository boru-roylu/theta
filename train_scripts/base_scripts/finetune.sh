#! /bin/bash -xe

. .env

TASK=finetune

if [ $# -lt 3 ]; then
    echo
    echo $0 [dataset] [model_name] [pos_type]
    echo
    exit 1
fi

DATASET=$1
MODEL_NAME=$2
POS_TYPE=$3

echo "====================================================================="
echo
echo "Start training: exp_name = $EXP_NAME; seed = $SEED"
echo
echo "====================================================================="

DEEPSPEED_JSON=./config/deepspeed/ds_config_zero3.json

MASTER_PORT=`python -c "print(int($$%9999+10000))"`
CACHE_DIR=`mktemp -d`
# First, if the TORCH_EXTENSIONS_DIR environment variable is set, it replaces <tmp>/torch_extensions and all extensions will be compiled into subfolders of this directory.
# https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load
export TORCH_EXTENSIONS_DIR=$CACHE_DIR/torch-extensions
OUTPUT_PARENT_DIR=$SCRATCH_DIR/exp/$DATASET/$EXP_NAME
NFS_OUTPUT_PARENT_DIR=$NFS_PARENT_DIR/exp/$DATASET/$EXP_NAME

OUTPUT_DIR=$OUTPUT_PARENT_DIR/$SUB_EXP_NAME
mkdir -p $OUTPUT_DIR

# For debugging purpose.
#--evaluation_strategy steps \
#--eval_steps $EVAL_STEPS \
#--save_steps $EVAL_STEPS \
#EVAL_STEPS=1

deepspeed --master_port $MASTER_PORT src/run_finetune_hibert.py \
    --task_name $TASK \
    --data_name $DATASET \
    --data_config_path $DATA_CONFIG_PATH \
    --do_train \
    --do_eval \
    --report_to tensorboard wandb \
    --seed $SEED \
    --model_path $MODEL_PATH \
    --model_name $MODEL_NAME \
    --coordinator_config_path $HIBERT_CONFIG_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epoch $NUM_TRAIN_EPOCH \
    --gradient_checkpoint true \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --remove_unused_columns false \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --metric_for_best_model $METRIC_FOR_BEST_MODEL \
    --greater_is_better true \
    --logging_strategy steps \
    --logging_steps 5 \
    --save_total_limit 1 \
    --dataloader_pin_memory \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --deepspeed $DEEPSPEED_JSON \
    --use_ks_model \
    --ddp_find_unused_parameters true \
    --warmup_ratio 0.1 \
    --copy_to_init_special_token_embeddings \
    --gradient_accumulation_steps 1 \
    --group_by_length \
    --length_column_name num_turns \
    --layerwise_learning_rate_decay 0.9 \
    --fp16 true \
    --fp16_backend auto \
    --fp16_opt_level O1 \
    --fp16_full_eval false \
    ${@:4} | tee $OUTPUT_DIR/stdout

echo "Copy files from $OUTPUT_PARENT_DIR to $NFS_OUTPUT_PARENT_DIR"
mkdir -p $NFS_OUTPUT_PARENT_DIR
python src/clean_up.py -d -e $OUTPUT_PARENT_DIR
cp -r $OUTPUT_PARENT_DIR/* $NFS_OUTPUT_PARENT_DIR

echo "====================================================================="
echo
echo "Finished training $EXP_NAME; seed = $SEED"
echo
echo "====================================================================="