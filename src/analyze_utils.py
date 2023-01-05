import numpy as np
import os

import pandas as pd
import sklearn
import torch

import clustering
import linear_assignment as la


embedding_name_to_easy_read_name = {
    'bert_mean_pooler_output': 'UttBert (Mean)',
    'bert_pooler_output': 'UttBert (CLS)',
    'ctx_pooler_output': 'ConvBert Output',
    'pooler_output': 'ConvBert Input',
}


def train(embeddings, num_clusters, cuda_device):
    use_gpu = torch.cuda.is_available()
    hidden_size = embeddings.shape[-1]
    kmeans = clustering.FaissClustering(
        dim=hidden_size,
        num_clusters=num_clusters,
        use_gpu=use_gpu,
        verbose=1,
        cuda_device=cuda_device)
    #kmeans.train(train_output[embedding_name][train_idxs])
    kmeans.train(embeddings)
    return kmeans


def save_embeddings(
    emb_dir,
    split,
    output,
    embedding_names=list(embedding_name_to_easy_read_name.keys())):

    split_emb_dir = os.path.join(emb_dir, split)
    os.makedirs(split_emb_dir, exist_ok=True)
    for k in embedding_names + ['dialog_lens']:
        with open(os.path.join(split_emb_dir, f'{k}.npy'), 'wb') as f:
            np.save(f, output[k])


def load_embeddings(emb_dir, split):
    split_emb_dir = os.path.join(emb_dir, split)
    ret = {}
    for k in embedding_name_to_easy_read_name.keys():
        path = os.path.join(split_emb_dir, f'{k}.npy')
        if not os.path.exists(path):
            continue
        with open(path, 'rb') as f:
            ret[k] = np.load(f)
    return ret


def get_label_dataframe(raw_ds, dialog_act_to_idx, da_col=None):

    turns = [t['turn'] for d in raw_ds['dialogue'] for t in d]

    dialog_idxs = [ex['dialog_idx'] for ex in raw_ds for _ in ex['dialogue']]
    turn_idxs = [t['turn_idx'] for d in raw_ds['dialogue'] for t in d]
    parties = [t['party'] for d in raw_ds['dialogue'] for _, t in enumerate(d)]

    assert (len(dialog_idxs) == len(turn_idxs) == len(turns) == len(parties))

    if da_col:
        dialog_acts = [t[da_col] 
             for d in raw_ds['dialogue'] for _, t in enumerate(d)]
        assert len(dialog_idxs) == len(dialog_acts)

        df = pd.DataFrame(
            zip(dialog_idxs, turn_idxs, turns, parties, dialog_acts),
            columns=['dialog_idx', 'turn_idx', 'turn', 'party', 'dialog_act'])
        df['dialog_act_idx'] = df['dialog_act'].replace(dialog_act_to_idx)
    else:
        df = pd.DataFrame(
            zip(dialog_idxs, turn_idxs, turns, parties),
            columns=['dialog_idx', 'turn_idx', 'turn', 'party'])
    return df


def get_clustering_assignment_to_label_mapping(labels, assignments):

    assert len(labels) == len(assignments)

    cm = sklearn.metrics.confusion_matrix(labels, assignments)
   
    indexes = la.linear_assignment(la.make_cost_m(cm))
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    mapping = dict(zip(js, range(len(js))))

    return mapping

