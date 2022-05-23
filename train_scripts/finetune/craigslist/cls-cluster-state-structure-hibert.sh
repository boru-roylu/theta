. .env

export NUM_TRAIN_EPOCH=4

export PER_DEVICE_TRAIN_BATCH_SIZE=16
export PER_DEVICE_EVAL_BATCH_SIZE=32
export LEARNING_RATE=1e-5
export WEIGHT_DECAY=1e-4
export METRIC_FOR_BEST_MODEL=dev_accuracy

DATASET=craigslist
MODEL_NAME=theta-cls-hibert
MODEL_BACKBONE=bert-base-uncased-${DATASET}-wwm

export MODEL_PATH=$NFS_PARENT_DIR/pretrained_models/${DATASET}/$MODEL_BACKBONE
export DATA_CONFIG_PATH=./config/data/${DATASET}.yaml
export HIBERT_CONFIG_PATH=./config/model/${MODEL_NAME}/cls-cluster-state-structure-hibert-absolute-pos-config-1layer-2head.json

mkdir -p /tmp/$USER
export SCRATCH_DIR=/tmp/$USER/

echo '==============================================================================='
echo 'Create data ...'
echo '==============================================================================='
bash ./data_scripts/${DATASET}.sh --task_name finetune

echo '==============================================================================='
echo 'Start training ...'
echo '==============================================================================='
POS_TYPE=absolute
export EXP_NAME=finetune/$MODEL_NAME-$POS_TYPE-pos_init_from_${MODEL_BACKBONE}_job_name-$SLURM_JOB_NAME

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
OUTPUT_PARENT_DIR=$NFS_PARENT_DIR/exp/$DATASET/$EXP_NAME
OUTPUT_DIR=$OUTPUT_PARENT_DIR/seed-$SEED

mkdir -p $OUTPUT_DIR

# For debugging purpose.
#--evaluation_strategy steps \
#--eval_steps $EVAL_STEPS \
#--save_steps $EVAL_STEPS \
#EVAL_STEPS=1
TASK=finetune

#python src/run_finetune_hibert.py \
#--gradient_checkpoint true \
deepspeed --master_port $MASTER_PORT src/run_finetune_hibert.py \
    --task_name $TASK \
    --data_name $DATASET \
    --data_config_path $DATA_CONFIG_PATH \
    --do_train \
    --do_eval \
    --report_to none \
    --model_path $MODEL_PATH \
    --model_name $MODEL_NAME \
    --coordinator_config_path $HIBERT_CONFIG_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epoch $NUM_TRAIN_EPOCH \
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
    --use_ks_model \
    --ddp_find_unused_parameters true \
    --warmup_ratio 0.1 \
    --copy_to_init_special_token_embeddings \
    --gradient_accumulation_steps 1 \
    --group_by_length \
    --length_column_name num_turns \
    --layerwise_learning_rate_decay 0.9 \
    --seed $SEED \
    --deepspeed $DEEPSPEED_JSON \
    --fp16 true \
    --fp16_backend auto \
    --fp16_opt_level O1 \
    --fp16_full_eval false \
    $@ | tee $OUTPUT_DIR/stdout

python src/clean_up.py -d -e $OUTPUT_PARENT_DIR

echo "====================================================================="
echo
echo "Finished training $EXP_NAME; seed = $SEED"
echo
echo "====================================================================="