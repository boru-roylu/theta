import collections
import copy
import itertools
import json
import os

import einops
import pandas as pd
import torch
import tqdm

import analyze_utils
import utils
import tensor_utils


def explode(df, column_name):
    return df.explode(['turn_index', column_name]).reset_index(drop=True)


def bi_state(arr):
    return list(zip(arr[:-1], arr[1:]))

        
def collapse_and_count(arr):
    collapsed_arr = []
    for element, repeats in itertools.groupby(arr):
        dic = {'element': element, 'count': len(list(repeats))}
        collapsed_arr.append(dic)
    return collapsed_arr


def collapse_sequence(arr):
    collapsed_arr = []
    for element, repeats in itertools.groupby(arr):
        collapsed_arr.append(element)
    return collapsed_arr


def duration(arr):
    collapsed_sequence_and_count = collapse_and_count(arr)
    total_len = sum([x['count'] for x in collapsed_sequence_and_count])
    dic = {}
    for x in collapsed_sequence_and_count:
        dic[x['element']] = x['count'] / total_len
    return dic


def convert_dict_key_to_string(dic):
    return {str(k): v for k, v in dic.items()}


def float_to_bool(dic):
    dic = copy.deepcopy(dic)
    for k, v in dic.items():
        dic[k] = int(bool(v))
    return dic


def get_turn_embeddings(
    model, dataset, collate_fn, batch_size, pop_keys, cuda_device):

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn)

    tensor_dict = collections.defaultdict(list)
    for batch in tqdm.tqdm(data_loader):
        batch_size = batch['input_ids'].size(0)

        for key in pop_keys:
            if key not in batch:
                continue
            batch.pop(key)

        utils.obj_to_device(batch, device=cuda_device)

        with torch.no_grad():
            output = model(**batch)

        attention_mask = batch['attention_mask'].detach().cpu()
        bert_last_hidden_state = output['last_hidden_state'].detach().cpu()
        bert_last_hidden_state = einops.rearrange(
            bert_last_hidden_state,
            '(bs nt) mnt hs -> bs nt mnt hs',
            bs=batch_size)
        bert_mean_pooler_output = tensor_utils.masked_mean(
            bert_last_hidden_state, attention_mask.unsqueeze(-1), 2)

        # attention_mask shape = (bt, nt).
        attention_mask = torch.clamp(attention_mask.sum(2), max=1).bool()

        tensor_dict['bert_mean_pooler_output'].append(bert_mean_pooler_output)
        tensor_dict['attention_mask'].append(attention_mask)

        # Releases GPU memory to avoid OOM.
        del output

    attention_mask = torch.cat(tensor_dict.pop('attention_mask'), dim=0)
    for k, v in tensor_dict.items():
        tensor = torch.cat(v, dim=0)
        tensor = tensor[attention_mask].numpy()
        tensor_dict[k] = tensor
    tensor_dict['dialog_lens'] = attention_mask.sum(1).numpy()
    return tensor_dict


def find_state_centers(
    splits,
    raw_ds,
    assignment_dir,
    state_assignment_dir,
    split_to_state_df,
    split_to_centers,
    topk_clusters_in_state=3):
    for split in splits:

        # Creates a joint dataframe for text, cluster, and state.
        assignment_path = os.path.join(assignment_dir, f'{split}.csv')

        dialog_idxs = \
            [ex['dialog_idx'] for ex in raw_ds[split] for _ in ex['dialogue']]
        turn_idxs = \
            [t['turn_idx'] for d in raw_ds[split]['dialogue'] for t in d]
        parties = [t['party'] for d in raw_ds[split]['dialogue'] for t in d]
        utts = [t['turn'] for d in raw_ds[split]['dialogue'] for t in d]
        text_df = pd.DataFrame(
            zip(dialog_idxs, turn_idxs, parties, utts),
            columns=['dialog_idx', 'turn_idx', 'party', 'utterance'])

        cluster_to_centers = split_to_centers[split]
        cluster_df = pd.read_csv(assignment_path)
        state_df = split_to_state_df[split]
        assert len(state_df) == len(cluster_df)

        structure_df = pd.merge(
            cluster_df,
            state_df,
            on=['dialog_idx', 'turn_idx'], suffixes=['_cluster', '_state'])
        df = pd.merge(text_df, structure_df, on=['dialog_idx', 'turn_idx'])

        # Finds state centers from clusters' centers for each party.
        state_to_centers = {}
        for state, g in df.groupby('assignment_state'):
        
            party_to_state_centers = {}
            for party, gg in g.groupby('party'):
                state_centers = []
                topk_major_clusters_in_state = \
                    gg['assignment_cluster'].value_counts().nlargest(
                        topk_clusters_in_state).index
        
                for cluster in topk_major_clusters_in_state:
                    centers = cluster_to_centers[str(cluster)]
                    centers = \
                        list(map(lambda x: {**x, 'cluster': cluster}, centers))
                    state_centers.extend(centers)
                party_to_state_centers[party] = state_centers
            state_to_centers[state] = party_to_state_centers
            
            center_path = \
                os.path.join(state_assignment_dir, f'{split}_center.json')
            with open(center_path, 'w') as f:
                json.dump(state_to_centers, f, indent=4)
                

def get_idxs_for_clustering(
    df, speaker1, speaker2, string_match_dialog_act_set=None):
    if string_match_dialog_act_set is not None:
        df = df[~df['dialog_act'].isin(string_match_dialog_act_set)]
    s1_subdf = df[df['party'] == speaker1]
    s2_subdf = df[df['party'] == speaker2]
    s1_idxs = s1_subdf.index.tolist()
    s2_idxs = s2_subdf.index.tolist()

    return s1_idxs, s2_idxs


def align_assignment(assignments, ref_df):
    # Converts assignment to label indices.
    labels = ref_df['dialog_act_idx']
    ref_assignments = assignments['dev']
    assignment_to_label = \
        analyze_utils.get_clustering_assignment_to_label_mapping(
            labels, ref_assignments)
    for split, assignment in assignments.items():
        assignments[split] = np.array(
            list(map(assignment_to_label.get, assignment)))


def clustering(splits, embeddings, kmeans_num_clusters, cuda_device):
    kmeans = analyze_utils.train(
        embeddings['train'], kmeans_num_clusters, cuda_device)
    distances, assignments = {}, {}
    for split in splits:
        distances[split], assignments[split] = kmeans.predict(embeddings[split])
    return distances, assignments


def add_ngram_features(df, sequence_column_name):
    df['uni_duration'] = df[sequence_column_name].apply(
        duration).apply(convert_dict_key_to_string)
    df['uni_indicator'] = df['uni_duration'].apply(float_to_bool)

    df['bi_sequence'] = df[sequence_column_name].apply(bi_state)
    df['bi_duration'] = df['bi_sequence'].apply(
        duration).apply(convert_dict_key_to_string)
    df['bi_indicator'] = df['bi_duration'].apply(float_to_bool)

    df['uni_duration_bi_indicator'] = [
        {**uni, **bi} for uni, bi in zip(
            df['uni_duration'], df['bi_indicator'])]
    df['uni_indicator_bi_indicator'] = [
        {**uni, **bi} for uni, bi in zip(
            df['uni_indicator'], df['bi_indicator'])]
    df['uni_duration_bi_duration'] = [
        {**uni, **bi} for uni, bi in zip(
            df['uni_duration'], df['bi_duration'])]
    return df