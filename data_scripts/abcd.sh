. .env

if [ $# -ne 2 ];
then
    echo $0 --task_name [task name]
    echo
    echo "task name can be finetune, pretrain or pretrain_bert."
    exit 0
fi

python src/create_data.py \
  --data_config_path ./config/data/abcd.yaml \
  --model_path $NFS_PARENT_DIR/pretrained_models/bert-base-uncased \
  --num_proc 8 \
  $@