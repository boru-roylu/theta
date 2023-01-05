import functools
import logging
import os
import sys

import datasets
import decouple 
import numpy as np
import transformers as tfs

import comm
import data_collator
import eval_utils
import model_init
import input_utils
import sklearn
import torch
import training_args
import trainer
import trainer_utils
import utils

logger = logging.getLogger()
logger.setLevel(logging.INFO)

NFS_DIR = decouple.config('NFS_PARENT_DIR')
SCRATCH_DIR = decouple.config('SCRATCH_DIR')
#SCRATCH_DIR = decouple.config('SCRATCH_PARENT_DIR')

# Gets training arguments.
parser = tfs.HfArgumentParser((training_args.TrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    train_args, = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    train_args, = parser.parse_args_into_dataclasses()

assert train_args.data_config_path

tokenizer = tfs.AutoTokenizer.from_pretrained(train_args.model_path)

data_config = utils.read_yaml(train_args.data_config_path)
task_name = train_args.task_name
dataset_dir = data_config['path']['dataset_dir'].format(scratch_dir=SCRATCH_DIR)

local_ds_dir = utils.get_dataset_dir_map(
    task_name, dataset_dir, train_args.model_path, train_args.debug_mode)

raw_ds = datasets.DatasetDict.load_from_disk(local_ds_dir['raw_ds'])

comm.wait_all()
columns = [
    'input_ids',
    'attention_mask',
    'num_turns',
    data_config['config']['label_name']]


# Adds strcuture features into dataset objects (raw_ds).
if (train_args.cluster_summary is not None
    or train_args.model_name == '${MODEL_NAME}'
    or train_args.model_name == 'theta-cls-hibert'):

    assert data_config['path'].get('assignment_dir', None)
    assert data_config['path'].get('state_assignment_dir', None)

    coordinator_config = tfs.AutoConfig.from_pretrained(
        train_args.coordinator_config_path)

    embedding_name = data_config['config']['embedding_name']
    embedding_name = train_args.embedding_name
    assert embedding_name is not None
    collapse_state_sequence = data_config['config']['collapse_state_sequence']
    num_clusters = train_args.num_clusters
    num_states = train_args.num_states
    assignment_dir = data_config['path']['assignment_dir'].format(nfs_dir=NFS_DIR)
    assignment_dir = os.path.join(
        assignment_dir, f'{embedding_name}/num_clusters_{num_clusters}')

    # Loads cluster ids.
    if getattr(coordinator_config, 'use_cluster_sequence_classifier', False):
        assert num_clusters is not None
        raw_ds = input_utils.add_assignments(
            raw_ds, assignment_dir, 'cluster', False, tokenizer=None)
        columns.append('cluster_input_ids')
        columns.append('cluster_attention_mask')

    # Loads state ids.
    if getattr(coordinator_config, 'use_state_sequence_classifier', False):
        # TODO(roylu): make this as arguments or other better way.
        num_splits = num_states - 3
        num_merging = 0
        assert num_clusters is not None
        assert num_states is not None
        state_assignment_dir = \
            data_config['path']['state_assignment_dir'].format(nfs_dir=NFS_DIR)
        state_assignment_dir = os.path.join(
            state_assignment_dir,
            f'{embedding_name}/num_clusters_{num_clusters}/num_splits_{num_splits}_num_states_{num_states}_num_merging_{num_merging}')
        raw_ds = input_utils.add_assignments(
            raw_ds, state_assignment_dir, 'state', collapse_state_sequence, tokenizer=None)
        columns.append('state_input_ids')
        columns.append('state_attention_mask')

ds = raw_ds
ds.set_format(type='torch', columns=columns)

model_init_map = {
    'theta-cls-hibert': model_init.theta_cls_hibert_model_init,
}

# Uses model_init to insure the reproducibility.
model_init_fn = model_init_map[train_args.model_name]
model_init_fn = functools.partial(
    model_init_fn, 
    train_args=train_args,
    tokenizer=tokenizer,
    data_config=data_config)

dc = data_collator.ConversationDataCollator(
    data_config['config']['label_name'], tokenizer.pad_token_id)

ks_post_process_function = \
    eval_utils.get_classification_finetuning_post_process_function(
        data_config['config']['num_labels'])

eval_splits = ['dev']

extra_log_function = \
    trainer_utils.get_extra_log_function(train_args, data_config)

our_trainer = trainer.MyTrainer(
    model_init=model_init_fn,
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
    compute_metrics=None,#eval_utils.compute_ks_metrics,
    ks_post_process_function=ks_post_process_function,
    pop_keys=data_config['finetune']['pop_keys'],
    keys_for_inference=data_config['finetune']['keys_for_inference'],
    ks_tokenizer=tokenizer,
    data_config=data_config,
    keep_logits=True)

if train_args.do_train:
    our_trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)