. .env

if [ $# -ne 2 ];
then
    echo $0 --task_name [task name]
    echo
    echo "task name can be `finetune`, `pretrain`, `pretrain_bert`."
    exit 1
fi

python src/create_data.py \
  --data_config_path ./config/data/craigslist.yaml \
  --model_path $NFS_PARENT_DIR/pretrained_models/bert-base-uncased \
  --num_proc 8 \
  $@