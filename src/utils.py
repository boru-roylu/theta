import os
import pandas as pd
import torch

import transformers as tfs
import yaml


def read_yaml(path):
    with open(path, 'r') as f:
        y = yaml.load(f, Loader=yaml.FullLoader)
    return y


def get_dataset_dir_map(task_name, dataset_dir, model_path, debug_mode):
    local_ds_subdir_names = []
    if debug_mode:
        local_ds_subdir_names += ['debug']
    model_base_name = \
        tfs.AutoConfig.from_pretrained(model_path, use_cache=False).model_type
    slurm_job_id = os.environ.get('SLURM_ARRAY_JOB_ID')
    slurm_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if slurm_job_id and slurm_task_id:
        slurm_dirname = f'job_{slurm_job_id}_task_{slurm_task_id}'
        local_ds_subdir_names += [slurm_dirname, model_base_name]
    else:
        local_ds_subdir_names += [model_base_name]
    subdirs = ['raw_ds']
    
    local_ds_dir = {}
    for subdir in subdirs:
        local_subdir_name = '/'.join(local_ds_subdir_names + [subdir])
        local_dir = os.path.join(dataset_dir, f'{task_name}/{local_subdir_name}')
        os.makedirs(local_dir, exist_ok=True)
        local_ds_dir[subdir] = local_dir
    return local_ds_dir


def obj_to_device(obj, device):
    # only put tensor into GPU
    if isinstance(obj, torch.Tensor):
        obj = obj.to(device)
    elif isinstance(obj, (list, tuple)):
        obj = list(obj)
        for v_i, v in enumerate(obj):
            obj[v_i] = obj_to_device(v, device)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = obj_to_device(v, device)
    return obj


def maybe_makedirs(target_dir, overwrite):
    if not overwrite:
        assert not os.path.exists(target_dir), f'{target_dir = } exists!'
    os.makedirs(target_dir, exist_ok=True)


def get_few_shot_sample_indices(labels, sample_percent, random_seed=42):
    df = pd.DataFrame(labels, columns=['label'])
    selected_indices = []
    unique_labels = df['label'].unique()
    sample_percent_per_class = sample_percent / len(unique_labels)
    num_samples_per_class = int(len(labels) * sample_percent_per_class)
    for unique_label in unique_labels:
        class_df = df[df['label'] == unique_label]
        assert len(class_df) >= num_samples_per_class
        indices = class_df.sample(
            n=num_samples_per_class, random_state=random_seed).index.tolist()
        selected_indices.extend(indices)

    total_indices = range() 
    return selected_indices