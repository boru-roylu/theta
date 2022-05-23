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

DATASET=craigslist
MODEL_NAME=bert
MODEL_BACKBONE=bert-base-uncased
TASK=pretrain_bert

export MODEL_PATH=$NFS_PARENT_DIR/pretrained_models/$MODEL_BACKBONE
export DATA_CONFIG_PATH=./config/data/${DATASET}.yaml

mkdir -p /tmp/$USER
export SCRATCH_DIR=/tmp/$USER/

echo '========================================================================='
echo 'Create data ...'
echo '========================================================================='
bash ./data_scripts/${DATASET}.sh --task_name pretrain_bert

echo '========================================================================='
echo 'Start training ...'
echo '========================================================================='
export SEED=42
export EXP_NAME=pretrain/${MODEL_NAME}_init_from_${MODEL_BACKBONE}_job_name-$SLURM_JOB_NAME

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


deepspeed --master_port $MASTER_PORT src/run_pretrain_bert.py \
    --task_name $TASK \
    --data_name $DATASET \
    --data_config_path $DATA_CONFIG_PATH \
    --do_train \
    --do_eval \
    --report_to tensorboard wandb \
    --seed $SEED \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --max_steps $MAX_STEPS \
    --gradient_checkpoint true \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --remove_unused_columns false \
    --save_strategy steps \
    --evaluation_strategy steps \
    --eval_steps $EVAL_STEPS \
    --save_steps $EVAL_STEPS \
    --metric_for_best_model $METRIC_FOR_BEST_MODEL \
    --greater_is_better true \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_total_limit 2 \
    --dataloader_pin_memory \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --deepspeed $DEEPSPEED_JSON \
    --use_ks_model \
    --save_ks_model \
    --ddp_find_unused_parameters true \
    --warmup_ratio 0.1 \
    --copy_to_init_special_token_embeddings \
    --group_by_length \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --fp16 true \
    --fp16_opt_level O1 \
    --fp16_backend auto \
    --fp16_full_eval false \
    $@ | tee $OUTPUT_DIR/stdout

python src/clean_up.py -d -e $OUTPUT_PARENT_DIR

echo "====================================================================="
echo
echo "Finished training $EXP_NAME; seed = $SEED"
echo
echo "====================================================================="