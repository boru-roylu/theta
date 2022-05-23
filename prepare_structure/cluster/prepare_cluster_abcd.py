import os
import sys
sys.path.append('src')

import gc

import copy
import datasets
import decouple
import json
import transformers as tfs
import torch
import tqdm

import analyze_utils
import clustering_utils
import data_collator
import modeling_bert
import utils

NFS_DIR = decouple.config('NFS_PARENT_DIR')

# # Loads data and initialize paths.
# ### Uses the corresponding script to create data first. (see file in data_sripts).

# Adjusts batch size according to your GPU size.
batch_size = 48
cuda_device = 0
data_name = 'abcd'
debug_mode = False

task_name = 'pretrain'
pos_type = 'absolute'

model_name = 'hibert'
data_config_path = f'./config/data/{data_name}.yaml'
coordinator_config_path = f'./config/model/theta-cls-hibert/cls-hibert-absolute-pos-config-1layer-2head.json'

data_config = utils.read_yaml(data_config_path)

pop_keys = ['mlm_labels', 'labels', 'sentence_masked_idxs']
model_path = data_config['path']['dapt_model_path'].format(nfs_dir=NFS_DIR)

assert os.path.isdir(model_path), model_path
emb_dir = data_config['path']['embedding_dir'].format(nfs_dir=NFS_DIR)
assignment_dir = data_config['path']['assignment_dir'].format(nfs_dir=NFS_DIR)

coordinator_config = tfs.AutoConfig.from_pretrained(coordinator_config_path)

os.makedirs(emb_dir, exist_ok=True)
os.makedirs(assignment_dir, exist_ok=True)

print(f'{model_path = }')
print(f'{coordinator_config_path = }')
print(f'{emb_dir = }')
print(f'{assignment_dir = }')

tokenizer = tfs.AutoTokenizer.from_pretrained(model_path)

model_config = tfs.AutoConfig.from_pretrained(model_path)

dataset_dir = data_config['path']['dataset_dir'].format(nfs_dir=NFS_DIR)

local_ds_dir = \
    utils.get_dataset_dir_map(task_name, dataset_dir, model_path, debug_mode)

if not os.path.exists(os.path.join(local_ds_dir['raw_ds'], 'dataset_dict.json')):
    raise ValueError('\n\nRun the script first: bash data_scripts/abcd.sh --task_name pretrain\n\n')

raw_ds = datasets.DatasetDict.load_from_disk(local_ds_dir['raw_ds'])
ds = copy.deepcopy(raw_ds)
ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])

print('='*100)
print('Show an example ...')
print('='*100)
idx = 20
for turn in raw_ds['train']['dialogue'][idx]:
    utt = turn['turn']
    party = turn['party']
    print(f'{party:>10}: {utt}')

# # Loads the model and extract embeddings.
model = modeling_bert.HierarchicalBertModel(
    config=model_config, coordinator_config=coordinator_config)
model.bert = tfs.AutoModel.from_pretrained(model_path)

_ = model.cuda(cuda_device)
_ = model.eval()

dc = data_collator.DataCollatorForWholeWordMaskAndWholeSentenceMask(
    tokenizer=tokenizer, 
    mlm_probability=0.0,
    msm_probability=0.0,
    max_num_turns=data_config['config']['max_num_turns'],
    min_num_valid_tokens=0,
    mask_whole_sentence=False)

# ### Extracts embeddings. (You can skip the step if you already did it.)
split_to_output = {}
for split in data_config['config']['splits']:
    print(f'Processing {split} set.')
    output = clustering_utils.get_turn_embeddings(
        model, ds[split], dc, batch_size, pop_keys, cuda_device)
    analyze_utils.save_embeddings(
        emb_dir, split, output, embedding_names=['bert_mean_pooler_output'])
    split_to_output[split] = output

# Cleans model on the GPU before using GPU-based clustering.
del model
gc.collect()
torch.cuda.empty_cache()

# # Loads extraced embeddings.
# If you already extracted embeddings, you can skip the process of extracting 
# embeddings above and directly load embeddings.
split_to_output = {}
for split in data_config['config']['splits']:
    split_to_output[split] = analyze_utils.load_embeddings(emb_dir, split)

# # Clusters turn embeddings.
# ABCD
# Sets the num of clusters.
num_clusters = 60

# Sets how many turns for each cluster you want to keep.
topk_center_turns = 20

dialog_act_to_full_name = {}
speaker1 = 'customer'
speaker2 = 'agent'
da_col = None
dialog_act_to_idx = None
string_match_dialog_act_set = None
embedding_names = ['bert_mean_pooler_output']


# Gets speaker indices.
split_to_df = {}
# Clusters two parties independently.
split_to_speaker1_idxs = {}
split_to_speaker2_idxs = {}
for split in data_config['config']['splits']:
    df = analyze_utils.get_label_dataframe(
        raw_ds[split], dialog_act_to_idx, da_col)
    df = df[df['party'] != '<start>']
    speaker1_idxs, speaker2_idxs = clustering_utils.get_idxs_for_clustering(
        df, speaker1, speaker2, string_match_dialog_act_set)
    split_to_df[split] = df
    split_to_speaker1_idxs[split] = speaker1_idxs
    split_to_speaker2_idxs[split] = speaker2_idxs

# # Gets assignments.
from IPython.display import display
max_num_turns = data_config['config']['max_num_turns']

# Shifts cluster if string matching is applied.
# Makes sure the string matching clustering uses the index starting from 0.
if string_match_dialog_act_set:
    kmeans_num_clusters = num_clusters - len(string_match_dialog_act_set)
else:
    kmeans_num_clusters = num_clusters

# Shifts cluster indices for the 2nd party.
assignment_idx_offset = num_clusters

for embedding_name in tqdm.tqdm(embedding_names):
    speaker1_embeddings = {
        split: split_to_output[split][embedding_name][split_to_speaker1_idxs[split]]
        for split in data_config['config']['splits']
    }
    speaker2_embeddings = {
        split: split_to_output[split][embedding_name][split_to_speaker2_idxs[split]]
        for split in data_config['config']['splits']
    }

    speaker1_distances, speaker1_assignments = clustering_utils.clustering(
        data_config['config']['splits'],
        speaker1_embeddings,
        kmeans_num_clusters,
        cuda_device)
    speaker2_distances, speaker2_assignments = clustering_utils.clustering(
        data_config['config']['splits'],
        speaker2_embeddings,
        kmeans_num_clusters,
        cuda_device)

    # Aligns with ground-truth if applicable.
    if da_col:
        ref_df = copy.deepcopy(split_to_df['dev'])
        ref_subdf = ref_df.loc[split_to_speaker1_idxs['dev']]
        clustering_utils.align_assignment(speaker1_assignments, ref_subdf)
        ref_subdf = ref_df.loc[split_to_speaker2_idxs['dev']]
        clustering_utils.align_assignment(speaker2_assignments, ref_subdf)

    sub_dir = os.path.join(
        assignment_dir, f'{embedding_name}/num_clusters_{num_clusters}')
    os.makedirs(sub_dir, exist_ok=True)

    for split in data_config['config']['splits']:
        df = split_to_df[split]
        df2 = copy.deepcopy(df)

        if da_col:
            # For string-matching DAs, we directly use ground-truth DA indices.
            df2['assignment'] = df2['dialog_act_idx']
        # For string-matching DAs, distance is 0. 
        df2['distance'] = [0] * len(df2)

        speaker1_idxs = split_to_speaker1_idxs[split]
        df2.loc[speaker1_idxs, 'assignment'] = speaker1_assignments[split]
        df2.loc[speaker1_idxs, 'distance'] = speaker1_distances[split]

        speaker2_idxs = split_to_speaker2_idxs[split]
        df2.loc[speaker2_idxs, 'assignment'] = speaker2_assignments[split] 
        df2.loc[speaker2_idxs, 'distance'] = speaker2_distances[split]

        all_speaker2_idxs = df2[df2['party'] == speaker2].index
        df2.loc[all_speaker2_idxs, 'assignment'] += assignment_idx_offset
        df2['assignment'] = df2['assignment'].astype(int)
        path = os.path.join(sub_dir, f'{split}.csv')
        df2[['dialog_idx', 'turn_idx', 'assignment']].to_csv(path, index=False)

        center_df = df2.groupby('assignment', group_keys=False).apply(
                lambda x: x.nsmallest(n=topk_center_turns, columns='distance'))

        assignment_to_centers = {}
        for assignment, g in center_df.groupby('assignment'):
            neighbors = []
            for _, row in g.iterrows():
                if assignment // num_clusters == 0:
                    party = speaker1
                else:
                    party = speaker2
                neighbor = {'party': party, 'turn': row['turn']}
                if da_col:
                    neighbor['dialog_act'] = row['dialog_act'],
                neighbors.append(neighbor)
            assignment_to_centers[assignment] = neighbors
        
        center_path = os.path.join(sub_dir, f'{split}_center.json')
        print(f'{"Save turns nearby centers":>30} {center_path:<50}')
        with open(center_path, 'w') as f:
            json.dump(assignment_to_centers, f, indent=4)