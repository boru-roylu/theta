import glob

import datasets
import decouple
import functools

import input_utils
import utils


NFS_DIR = decouple.config("NFS_PARENT_DIR")


def bert_wwm_pretrain(ds, data_config, tokenizer, num_proc):
    num_turns = data_config["pretrain_bert"]["num_turns_per_chunk"]
    slide = data_config["pretrain_bert"]["slide"]
    max_seq_length = data_config["pretrain_bert"]["max_seq_length"]
    chunk_fn = input_utils.get_chunk_fn(num_turns=num_turns, slide=slide)
    ds = ds.map(chunk_fn, batched=True, num_proc=num_proc)

    encode_fn = input_utils.get_pretrain_encode_fn(
        tokenizer, max_seq_length=max_seq_length
    )
    ds = ds.map(encode_fn, num_proc=num_proc)
    return ds


def craigslist_pretrain_and_finetune(
    ds, data_config, tokenizer, num_proc, is_remove_cue_turn=False
):
    config = data_config["config"]

    if is_remove_cue_turn:
        ds = ds.map(input_utils.remove_cue_turn, num_proc=num_proc)

    encode_fn = input_utils.get_encode_fn(
        tokenizer, max_seq_length=config["max_seq_length_per_turn"]
    )
    ds = ds.map(encode_fn, num_proc=num_proc, batched=True)

    return ds


def abcd_pretrain_and_finetune(ds, data_config, tokenizer, num_proc):
    config = data_config["config"]
    ds = ds.map(input_utils.remove_action_turns, num_proc=num_proc)

    encode_fn = input_utils.get_encode_fn(
        tokenizer, max_seq_length=config["max_seq_length_per_turn"]
    )
    ds = ds.map(encode_fn, num_proc=num_proc, batched=True)

    return ds


function_map = {
    "abcd": {
        "pretrain_bert": bert_wwm_pretrain,
        "pretrain": abcd_pretrain_and_finetune,
        "finetune": abcd_pretrain_and_finetune,
    },
    "craigslist": {
        "pretrain_bert": bert_wwm_pretrain,
        "pretrain": craigslist_pretrain_and_finetune,
        "finetune": functools.partial(
            craigslist_pretrain_and_finetune, is_remove_cue_turn=True
        ),
    },
    "tennis": {
        "pretrain_bert": bert_wwm_pretrain,
        "pretrain": abcd_pretrain_and_finetune,
        "finetune": abcd_pretrain_and_finetune,
    },
    "wsdm": {
        "pretrain_bert": bert_wwm_pretrain,
        "pretrain": abcd_pretrain_and_finetune,
        "finetune": abcd_pretrain_and_finetune,
    },
}


def load_datasets(
    data_config_path,
    task_name,
    tokenizer=None,
    num_proc=1,
    splits=["train", "dev", "test"],
    features=None,
    debug=False,
):
    data_config = utils.read_yaml(data_config_path)

    input_pattern_map = {
        k: v.format(nfs_dir=NFS_DIR)
        for k, v in data_config["path"]["input_pattern_map"].items()
    }
    print(input_pattern_map)
    cache_dir = data_config["path"]["cache_dir"].format(nfs_dir=NFS_DIR)

    # Loads dialog dataset.
    data_files = {}
    for split in splits:
        input_pattern = input_pattern_map[split]
        input_files = glob.glob(input_pattern)
        data_files[split] = input_files

    ds = datasets.load_dataset(
        "json", features=features, data_files=data_files, cache_dir=cache_dir
    )

    if debug:
        for k in ds.keys():
            ds[k] = ds[k].select(range(3))

    func = function_map[data_config["data_name"]][task_name]
    ds = func(ds, data_config, tokenizer, num_proc)

    def add_dialog_idx(example, idx):
        example['dialog_idx'] = idx
        return example

    if 'dialog_idx' not in ds['train']:
        ds = ds.map(add_dialog_idx, num_proc=num_proc, with_indices=True)

    return ds
