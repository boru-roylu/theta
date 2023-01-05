import logging
import os
import sys

import decouple
import datasets
import transformers as tfs

import comm
import eval_utils
import training_args
import trainer_utils
import trainer
import utils

logger = logging.getLogger()
logger.setLevel(logging.INFO)
datasets.logging.set_verbosity(datasets.logging.INFO)

NFS_DIR = decouple.config('NFS_PARENT_DIR')
SCRATCH_DIR = decouple.config('SCRATCH_DIR')

# Gets training arguments.
parser = tfs.HfArgumentParser((training_args.TrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    train_args, = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    train_args, = parser.parse_args_into_dataclasses()

assert train_args.data_config_path, f'{train_args.data_config_path = }'

tokenizer = tfs.AutoTokenizer.from_pretrained(train_args.model_path)

task_name = train_args.task_name 
data_config = utils.read_yaml(train_args.data_config_path)
dataset_dir = data_config['path']['dataset_dir'].format(scratch_dir=SCRATCH_DIR)

local_ds_dir = utils.get_dataset_dir_map(
    task_name, dataset_dir, train_args.model_path, train_args.debug_mode)

raw_ds = datasets.DatasetDict.load_from_disk(local_ds_dir['raw_ds'])

comm.wait_all()
columns = ['input_ids', 'attention_mask']

ds = raw_ds
ds.set_format(type='torch', columns=columns)

# Uses model_init to insure the reproducibility.
def model_init():
    model = tfs.AutoModelForMaskedLM.from_pretrained(train_args.model_path)
    return model

tokenizer = tfs.AutoTokenizer.from_pretrained(train_args.model_path)
dc = tfs.DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm_probability=0.15)

ks_post_process_function = \
    eval_utils.get_pretraining_post_process_function(tokenizer, 'labels')
extra_log_function = \
    trainer_utils.get_extra_log_function(train_args, data_config)

eval_splits = ['dev']

our_trainer = trainer.MyTrainer(
    model_init=model_init,
    args=train_args,
    train_dataset=ds['train'] if train_args.do_train else None,
    eval_dataset=ds['dev'] if train_args.do_eval else None,
    test_dataset=ds.get('test', None) if train_args.do_eval else None,
    eval_metric_key_prefix='dev',
    test_metric_key_prefix='test',
    eval_datasets_at_end=ds,
    eval_splits_on_best_dev_at_end=eval_splits,
    extra_log_function=extra_log_function,
    train_data_collator=dc,
    eval_data_collator=dc,
    compute_metrics=None,
    ks_post_process_function=ks_post_process_function,
    pop_keys=data_config['pretrain_bert']['pop_keys'],
    keys_for_inference=data_config['pretrain_bert']['keys_for_inference'],
    ks_tokenizer=tokenizer,
    data_config=data_config,
    keep_logits=False)

if train_args.do_train:
    our_trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)