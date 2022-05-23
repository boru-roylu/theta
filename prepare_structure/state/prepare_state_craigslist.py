import sys
sys.path.append('src')

import utils
import clustering_utils
import clustering
import pandas as pd
import json
import decouple
import datasets
import collections
import copy
import os


NFS_DIR = decouple.config('NFS_PARENT_DIR')

data_name = 'craigslist'
num_clusters = 14
embedding_name = 'bert_mean_pooler_output'
cuda_device = 0
debug_mode = False
n_jobs = 8
overwrite = True

task_name = 'pretrain'
pos_type = 'absolute'

model_name = 'hibert'
data_config_path = f'./config/data/{data_name}.yaml'

data_config = utils.read_yaml(data_config_path)

hmm_config_path = f'./config/hmm/{data_name}.yaml'
hmm_config = utils.read_yaml(hmm_config_path)

model_path = data_config['path']['dapt_model_path'].format(nfs_dir=NFS_DIR)
assert os.path.isdir(model_path), model_path

assignment_parent_dir = data_config['path']['assignment_dir'].format(
    nfs_dir=NFS_DIR)
assignment_dir = os.path.join(
    assignment_parent_dir, f'{embedding_name}/num_clusters_{num_clusters}')

state_assignment_parent_dir = \
    data_config['path']['state_assignment_dir'].format(nfs_dir=NFS_DIR)
os.makedirs(state_assignment_parent_dir, exist_ok=True)

print(f'{model_path = }')
print(f'{assignment_dir = }')
print(f'{state_assignment_parent_dir = }')

dataset_dir = data_config['path']['dataset_dir'].format(nfs_dir=NFS_DIR)

local_ds_dir = utils.get_dataset_dir_map(
    task_name, dataset_dir, model_path, debug_mode)

raw_ds = datasets.DatasetDict.load_from_disk(local_ds_dir['raw_ds'])
ds = copy.deepcopy(raw_ds)
ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])

split_to_cluster_seq_df = {}
split_to_centers = {}
for split in data_config['config']['splits']:
    assignment_path = os.path.join(assignment_dir, f'{split}.csv')
    center_path = os.path.join(assignment_dir, f'{split}_center.json')
    parties = [turn['party']
               for dialogue in raw_ds[split]['dialogue'] for turn in dialogue]
    df = pd.read_csv(assignment_path)
    df['party'] = parties
    cluster_seq_df = df.groupby('dialog_idx')['assignment'].apply(
        list).reset_index(name='cluster_sequence')
    cluster_seq_df['party_sequence'] = df.groupby('dialog_idx')['party'].apply(
        list)
    cluster_seq_df['turn_idx'] = df.groupby('dialog_idx')['turn_idx'].apply(
        list)

    split_to_cluster_seq_df[split] = cluster_seq_df

    with open(center_path, 'r') as f:
        split_to_centers[split] = json.load(f)

init_min_seq_len = hmm_config['init_min_seq_len']
topk_edges = hmm_config['topk_edges']
num_init_states = hmm_config['num_init_states']
max_iterations = hmm_config['max_iterations']
num_state_splits = hmm_config['num_state_splits']
topk_clusters = 3
topk_utterances_per_cluster = 3
topk_clusters_in_state = 3
index_to_party = {v: k for k,
                  v in data_config['config']['party_to_index'].items()}
speaker1 = index_to_party[0]
speaker2 = index_to_party[1]

# %%
cluster_to_utterances = {}
for cluster, neighbors in split_to_centers['train'].items():
    cluster_to_utterances[cluster] = \
        [f"{n['party']}: {n['turn']}"
         for n in neighbors][:topk_utterances_per_cluster]
assert len(cluster_to_utterances) == num_clusters * 2

# %%
train_sequences = split_to_cluster_seq_df['train']['cluster_sequence'].tolist()
dev_sequences = split_to_cluster_seq_df['dev']['cluster_sequence'].tolist()
cluster_vocab = list(set(sum(train_sequences+dev_sequences, [])))

# # Trains HMM and apply state-splitting algorithm to build topologies.
# Initializes 3-state left-to-right topology.
hmm = clustering.DiscreteSequenceClustering(
    init_min_seq_len,
    cluster_vocab,
    num_init_states=num_init_states,
    topk_edges=topk_edges)

hmm.build(train_sequences)
#hmm.save_model(checkpoint_path)

curr_split = 0
state_assignment_dir = os.path.join(
    state_assignment_parent_dir,
    f'{embedding_name}/num_clusters_{num_clusters}/'
    f'num_splits_{hmm.num_splits}_num_states_{hmm.num_states}_num_merging_{hmm.num_merging}')
utils.maybe_makedirs(state_assignment_dir, overwrite)

print(f'Total # splits = {num_state_splits}')
num_states_to_model = {}
for curr_split in range(1, num_state_splits+1):

    print('*'*50)
    print(f'Current split     = {curr_split}.')
    print(f'Current # states  = {hmm.num_states}.')
    print(f'Current # merging = {hmm.num_merging}.')
    print(f'Current # splits  = {hmm.num_splits}.')
    print('*'*50)

    print('='*50)
    print(f'Start splitting.')
    print('='*50)
    hmm.split_and_train(train_sequences, train_sequences,
                        max_iterations, n_jobs)

    state_assignment_parent_dir,
    state_assignment_dir = os.path.join(
        state_assignment_parent_dir,
        f'{embedding_name}/num_clusters_{num_clusters}/'
        f'num_splits_{hmm.num_splits}_num_states_{hmm.num_states}_num_merging_{hmm.num_merging}')
    utils.maybe_makedirs(state_assignment_dir, overwrite)
    checkpoint_path = os.path.join(state_assignment_dir, f'hmm.pkl')

    hmm.save_model(checkpoint_path)
    num_states_to_model[hmm.num_states] = copy.deepcopy(hmm)
    print(f'Save hmm to {checkpoint_path = }')
    print('\n\n')

    title = state_assignment_dir

    # Plots figure and predicts state sequneces.
    hmm = clustering.ClusteringBase.from_pretrained(checkpoint_path)
    g = clustering.plot_hmm(
        hmm, topk_clusters, cluster_to_utterances, title=state_assignment_dir)
    image_path = os.path.join(state_assignment_dir, 'hmm')
    g.render(image_path, format='pdf')

    split_to_state_df = {}
    for split, seq_df in split_to_cluster_seq_df.items():
        sequences = seq_df['cluster_sequence'].tolist()
        seq_df['state_sequence'] = list(map(hmm.predict, sequences))
        df = seq_df[['dialog_idx', 'turn_idx', 'state_sequence']].explode(
            ['turn_idx', 'state_sequence']).reset_index(drop=True)
        df.rename(columns={'state_sequence': 'assignment'}, inplace=True)
        split_to_state_df[split] = df
        path = os.path.join(state_assignment_dir, f'{split}.csv')
        df.to_csv(path, index=False)
        print(f'Saves state assignment to {path}')

    clustering_utils.find_state_centers(
        data_config['config']['splits'],
        raw_ds,
        assignment_dir,
        state_assignment_dir,
        split_to_state_df,
        split_to_centers,
        topk_clusters_in_state=3)