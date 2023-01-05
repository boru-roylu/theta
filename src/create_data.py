import argparse

import decouple 
import transformers as tfs

import input_pipeline
import utils

SCRATCH_DIR = decouple.config('SCRATCH_DIR', default=None)
if not SCRATCH_DIR:
    SCRATCH_DIR = decouple.config('SCRATCH_PARENT_DIR')


parser = argparse.ArgumentParser(prog='Xtrayce')
parser.add_argument(
    '--data_config_path',
    type=str,
    required=True,
    help='Path of data onfiguration.')

parser.add_argument(
    '--task_name',
    type=str,
    required=True,
    choices=['pretrain', 'pretrain_bert', 'finetune'],
    help='Task name.')

parser.add_argument(
    '--model_path',
    type=str,
    required=True,
    help='Path of directory of the pretrained model.')

parser.add_argument(
    '--debug_mode',
    help='Create smaller HF dataset object for debugging.',
    action='store_true')

parser.add_argument(
    '--num_proc',
    default=10,
    type=int,
    help='# processes during preprocessing.')
    
args = parser.parse_args()

data_config = utils.read_yaml(args.data_config_path)
data_name = data_config['data_name']
splits = data_config['config']['splits']
dataset_dir = data_config['path']['dataset_dir'].format(scratch_dir=SCRATCH_DIR)

tokenizer = tfs.AutoTokenizer.from_pretrained(args.model_path)

local_ds_dir = utils.get_dataset_dir_map(
    args.task_name, dataset_dir, args.model_path, args.debug_mode)

raw_ds = input_pipeline.load_datasets(
    args.data_config_path,
    args.task_name,
    tokenizer=tokenizer,
    num_proc=args.num_proc,
    splits=splits,
    debug=args.debug_mode)

raw_ds.save_to_disk(local_ds_dir['raw_ds'])

print(f"Path: {local_ds_dir['raw_ds']}")