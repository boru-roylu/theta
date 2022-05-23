import os
import re

import collections
import pandas as pd
import torch

import clustering_utils


def remove_action_turns(example):
    dialogue = example['dialogue']
    new_dialogue = []
    for utt in dialogue:
        if utt['party'] == 'action':
            continue
        new_dialogue.append(utt)
    example['dialogue'] = new_dialogue
    return example


def remove_cue_turn(example):
    example['dialogue'] = example['dialogue'][:-1]
    return example


def get_encode_fn(
    tokenizer, max_seq_length=64, max_num_turns=256, strip_punctuation=False):

    def encode(examples):
        dialogues = examples['dialogue']

        new_dialogues = []
        for d in dialogues:
            new_dialogues.append(d[:max_num_turns])
    
        num_turns = [len(d) for d in new_dialogues]

        if strip_punctuation:
            func = lambda x: re.sub(r'[^\w\s]', '', x)
        else:
            func = lambda x: x

        utterances = [
            f"{utt['party']}: {func(utt['turn'])}" \
                for d in new_dialogues for utt in d]

        inputs = tokenizer(utterances,
                           max_length=max_seq_length,
                           truncation=True,
                           return_token_type_ids=False,
                           padding='max_length',
                           return_tensors='pt')

        if torch.any(inputs['input_ids'] >= 30522):
            assert False

        examples['input_ids'] = list(
            torch.split(inputs['input_ids'], num_turns))
        examples['attention_mask'] = list(
            torch.split(inputs['attention_mask'], num_turns))

        examples['num_turns'] = num_turns
        return examples

    return encode


def get_pretrain_encode_fn(tokenizer, max_seq_length=128):

    def encode(example):
        dialogue = example['dialogue']

        utt = dialogue[0]
        utterance = f"{utt['party']}: {utt['turn']}"
        for utt in dialogue[1:]:
            utterance += f"[SEP] {utt['party']}: {utt['turn']}"

        inputs = tokenizer(utterance,
                           max_length=max_seq_length,
                           truncation=True,
                           return_token_type_ids=False,
                           padding='max_length',
                           return_tensors='pt')

        example['input_ids'] = inputs['input_ids'][0]
        example['attention_mask'] = inputs['attention_mask'][0]
        return example

    return encode


def get_chunk_fn(num_turns, slide):
    def chunk_dialog(examples):
        dialogs = examples.pop('dialogue')
        keys = list(examples.keys())

        new_examples = collections.defaultdict(list)
        for example_idx, dialog in enumerate(dialogs):
            for i in range(0, len(dialog), slide):
                chunk = dialog[i:i+num_turns]
                new_examples['dialogue'].append(chunk)
                for key in keys:
                    new_examples[key].append(examples[key][example_idx])
        return dict(new_examples)
    return chunk_dialog


def get_assignment_func(
    dialog_idx_to_assignments, col, collapse, tokenizer=None):
    def func(example):
        dialog_idx = example['dialog_idx']
        assignments = dialog_idx_to_assignments[dialog_idx]

        if collapse:
            assignments = clustering_utils.collapse_sequence(assignments)

        if tokenizer is None:
            example[f'{col}_input_ids'] = assignments
            example[f'{col}_attention_mask'] = [1] * len(assignments)
        else:
            assignments = ' '.join(map(str, assignments))
            inputs = tokenizer(assignments)
            example[f'{col}_input_ids'] = inputs['input_ids']
            example[f'{col}_attention_mask'] = inputs['attention_mask']
        
        return example
    return func


def add_assignments(raw_ds, assignment_dir, col, collapse=False, tokenizer=None):
    for split in raw_ds.keys():
        path = os.path.join(assignment_dir, f'{split}.csv')
        df = pd.read_csv(path)
    
        df = df.groupby('dialog_idx')['assignment'].apply(list).reset_index()
        dialog_idx_to_assignments = \
            dict(zip(df['dialog_idx'], df['assignment']))
        dialog_idx_to_assignments
        func = get_assignment_func(
            dialog_idx_to_assignments, col, collapse, tokenizer=tokenizer)
        raw_ds[split] = raw_ds[split].map(func)
    return raw_ds 