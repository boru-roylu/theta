from typing import List, Union, Any, Dict, Tuple
import warnings
import random

import einops
import torch
import torch.nn.functional as F
import transformers as tfs
from transformers.tokenization_utils_base import BatchEncoding
from transformers.data.data_collator import _torch_collate_batch, tolist

import tensor_utils


class ConversationDataCollator:
    def __init__(self, label_name, pad_token_id):
        self.label_name = label_name
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        first = features[0]
        keys = set(first.keys())

        batch = {}
        if 'example_id' in first:
            example_ids = [f['example_id'] for f in features]
            batch['example_ids'] = torch.stack(example_ids)

        if self.label_name in keys:
            keys.remove(self.label_name)
            labels = [f[self.label_name] for f in features]
            batch['labels'] = torch.stack(labels)

        if 'num_turns' in keys:
            num_turns = [f['num_turns'] for f in features]
            batch['num_turns'] = torch.stack(num_turns)

        for k in keys:
            if k in ('example_id', 'labels', 'num_turns'):
                continue

            if k == 'input_ids':
                pad_token_id = self.pad_token_id
            else:
                pad_token_id = 0

            max_seq_len = max([len(f[k]) for f in features])
            to_batch = []
            for f in features:
                #if isinstance(f[k], list):
                if f[k].ndim == 2:
                    # 2D tensor (e.g. input_ids and attention_mask).
                    t = f[k]
                    seq_len = t.size(0)
                    pad_len = max_seq_len - seq_len
                    t = F.pad(t, (0, 0, 0, pad_len), value=pad_token_id)
                elif f[k].ndim == 1:
                    # 1D tensor (e.g. cluster_ids and state_ids).
                    t = f[k]
                    seq_len = t.size(0)
                    pad_len = max_seq_len - seq_len
                    t = F.pad(t, (0, pad_len), value=pad_token_id)
                else:
                    raise ValueError(f'Dimension {f[k].ndim} is not supported for `{k}`.')
                to_batch.append(t)
            batch[k] = torch.stack(to_batch)
        return batch

        
class DataCollatorForWholeWordMaskAndWholeSentenceMask(tfs.DataCollatorForWholeWordMask):
    def __init__(
        self,
        tokenizer,
        mlm_probability,
        msm_probability,
        max_num_turns=None,
        min_num_valid_tokens=None,
        mask_whole_sentence=True,
        labels_name=None,
        is_train=False,
        device='cpu',
        seed=42):
        super().__init__(tokenizer, mlm_probability)
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.msm_probability = msm_probability
        self.max_num_turns = max_num_turns
        self.min_num_valid_tokens = min_num_valid_tokens
        self.mask_whole_sentence = mask_whole_sentence
        self.labels_name = labels_name
        self.is_train = is_train
        self.seed = seed
        self.generator = None
        if not self.is_train:
            self.generator = torch.Generator(device)
            self.generator.manual_seed(seed)
            self.rng = random.Random(seed)

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        if not isinstance(self.tokenizer, (tfs.models.bert.BertTokenizer, tfs.models.bert.BertTokenizerFast)):
            warnings.warn(
                "DataCollatorForWholeWordMask is only suitable for BertTokenizer-like tokenizers. "
                "Please refer to the documentation for more information."
            )

        cand_indexes = []
        for i, token in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        if not self.is_train:
            self.rng.seed(self.seed)
            self.rng.shuffle(cand_indexes)
        else:
            random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def torch_mask_tokens(self, inputs: Any, mask_labels: Any) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """

        if not self.is_train:
            self.generator.manual_seed(self.seed)

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
                " --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8), generator=self.generator).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5), generator=self.generator).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, generator=self.generator)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def mlm_torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            input_ids = [e["input_ids"] for e in examples]
        else:
            input_ids = examples
            examples = [{"input_ids": e} for e in examples]

        batch_input = _torch_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

        mask_labels = []
        for e in examples:
            ref_tokens = []
            for id in tolist(e["input_ids"]):
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            # For Chinese tokens, we need extra inf to mark sub-word, e.g [喜,欢]-> [喜，##欢]
            if "chinese_ref" in e:
                ref_pos = tolist(e["chinese_ref"])
                len_seq = len(e["input_ids"])
                for i in range(len_seq):
                    if i in ref_pos:
                        ref_tokens[i] = "##" + ref_tokens[i]
            mask_labels.append(self._whole_word_mask(ref_tokens))
        mlm_mask = _torch_collate_batch(
            mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        inputs, labels = self.torch_mask_tokens(batch_input, mlm_mask)

        return {"input_ids": inputs, "labels": labels}

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # examples contains only input_ids, attention_mask, hidden_units

        dialogue_lens = torch.tensor(
            [len(ex['input_ids']) for ex in examples], dtype=torch.long)

        # If max_num_turns is not provided, use the max # turns of the current 
        # batch to pad.
        if self.max_num_turns:
            max_num_turns = self.max_num_turns
        else:
            max_num_turns = dialogue_lens.max()
        num_dialogues = len(examples)
        turn_mask = torch.arange(max_num_turns) < dialogue_lens.view(-1, 1)

        # target_sentence_idxs shape = (nd, mnt).
        # sentence_masked_indices shape = (nd, mnt).
        target_sentence_idxs, sentence_masked_indices = \
            self._mask_sentences(num_dialogues, max_num_turns, ~turn_mask)

        def pad_turns(tensor_list, pad_value):
            #tensor_list = torch.tensor_split(tensor, dialogue_lens[:-1])
                    #torch.stack(x), max_num_turns, pad_value, dim=0)
            tensor_list = [
                tensor_utils.pad_tensor(x, max_num_turns, pad_value, dim=0)
                for x in tensor_list]
            return torch.cat(tensor_list)

        input_ids_list = [ex['input_ids'] for ex in examples]
        attention_mask_list = [ex['attention_mask'] for ex in examples]
        input_ids = pad_turns(input_ids_list, self.tokenizer.pad_token_id)
        attention_mask = pad_turns(attention_mask_list, 0)

        if self.labels_name == 'hidden_units':
            assert 'hidden_units' in examples[0]
            hidden_units_list = [ex['hidden_units'] for ex in examples]
            hidden_units = torch.stack(
                [tensor_utils.pad_tensor(hu_b, max_num_turns, -100, dim=-1)
                for hu_b in hidden_units_list])
            gold_hidden_units_list = [ex['gold_hidden_units'] for ex in examples]
            gold_hidden_units = torch.stack(
                [tensor_utils.pad_tensor(hu_b, max_num_turns, -100, dim=-1)
                for hu_b in gold_hidden_units_list])
            #if self.is_train:
            #    hidden_units.masked_fill_(~sentence_masked_indices, -100)

        # Makes sure we get decoder_labels from input_ids before MLM.
        decoder_labels = input_ids[target_sentence_idxs.masked_fill(
                target_sentence_idxs == -100, 0)]

        input_ids = [x.view(-1) for x in torch.split(input_ids, 1)]
        attention_mask = [x.view(-1) for x in torch.split(attention_mask, 1)]
        examples = []
        for inp_ids, attn_mask in zip(input_ids, attention_mask):
            examples.append({'input_ids': inp_ids, 'attention_mask': attn_mask})

        # Applies WWM.
        ret = self.mlm_torch_call(examples)

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(
                val, already_has_special_tokens=True) for val in input_ids
        ]
        special_tokens_mask = torch.tensor(
            special_tokens_mask, dtype=torch.bool)
        special_tokens_mask = einops.rearrange(
            special_tokens_mask,
            '(bs mnt) nt -> bs mnt nt',
            mnt=max_num_turns)

        input_ids = ret['input_ids']
        mlm_labels = ret['labels']
        attention_mask = input_ids != self.tokenizer.pad_token_id

        attention_mask = einops.rearrange(
            attention_mask, '(bs mnt) nt -> bs mnt nt', mnt=max_num_turns)
        mlm_labels = einops.rearrange(
            mlm_labels, '(bs mnt) nt -> bs mnt nt', mnt=max_num_turns)

        if self.min_num_valid_tokens:
            # Only makes long sentences as targets.
            # num_valid_tokens shape = (nd, mnt).
            num_valid_tokens = attention_mask.sum(2)
            short_sentence_mask = num_valid_tokens < self.min_num_valid_tokens
            target_sentence_idxs = target_sentence_idxs.masked_fill(
                short_sentence_mask, -100)
            sentence_masked_indices = sentence_masked_indices.masked_fill(
                short_sentence_mask, 0)

            #short_sentence_mask = short_sentence_mask.unsqueeze(-1)
            #labels = labels.masked_fill(short_sentence_mask, -100)

        num_tokens = input_ids.size(-1)
        helper = einops.repeat(
            target_sentence_idxs, 'nd mnt -> nd mnt nt', nt=num_tokens)

        # decoder_input_ids and decoder_labels are almost the same except 
        # decoder_labels uses -100 for padding.
        decoder_input_ids = decoder_labels.masked_fill(helper == -100, 0)
        #print(f"{self.tokenizer.decode(decoder_input_ids[0][a[0]]) = }")
        decoder_attention_mask = (decoder_input_ids != 0).long()
        decoder_labels = decoder_labels.masked_fill(helper == -100, -100) 
        decoder_labels = decoder_labels.masked_fill(
            decoder_labels == self.tokenizer.pad_token_id, -100)
        
        input_ids = einops.rearrange(
            input_ids, '(bs mnt) nt -> bs mnt nt', mnt=max_num_turns)

        if self.mask_whole_sentence:
            # Expands to token dim.
            expanded_sentence_masked_indices = einops.repeat(
                sentence_masked_indices, 'nd mnt -> nd mnt nt', nt=num_tokens)

            # Masks out entire sentences but keeping [CLS] and [SEP].
            expanded_sentence_masked_indices = \
                expanded_sentence_masked_indices & (~special_tokens_mask)
            input_ids = input_ids.masked_fill(
                expanded_sentence_masked_indices, self.tokenizer.mask_token_id)

            # NOTE (roylu): we should not masked out labels because we want the model to use context to predict the masked sentences.
            #mlm_labels = mlm_labels.masked_fill(
            #    expanded_sentence_masked_indices, -100)

        ret['input_ids'] = input_ids
        ret['mlm_labels'] = mlm_labels
        ret['attention_mask'] = attention_mask
        ret['sentence_masked_idxs'] = sentence_masked_indices
        if self.labels_name == 'decoder_labels':
            ret['decoder_input_ids'] = decoder_input_ids
            ret['decoder_attention_mask'] = decoder_attention_mask
            ret['labels'] = decoder_labels
        elif self.labels_name == 'hidden_units':
            ret['labels'] = hidden_units
            ret['gold_labels'] = gold_hidden_units

        return ret

    def _mask_sentences(self, num_dialogues, max_num_sentences, special_tokens_mask):
        if not self.is_train:
            self.generator.manual_seed(self.seed)

        target_sentence_indices = torch.arange(num_dialogues * max_num_sentences)
        target_sentence_indices = target_sentence_indices.view(num_dialogues, max_num_sentences)
        shape = target_sentence_indices.shape
        
        probability_matrix = torch.full(shape, self.msm_probability)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        masked_indices = torch.bernoulli(
            probability_matrix, generator=self.generator).bool()
        
        # 80% of the time, we keep generating the original masked sentence.
        indices_replaced = torch.bernoulli(
            torch.full(shape, 0.8), generator=self.generator).bool() & masked_indices
        
        # 10% of the time, we generate a random sentence in the batch for the masked sentence.
        indices_random = torch.bernoulli(
            torch.full(shape, 0.5), generator=self.generator).bool() & masked_indices & ~indices_replaced
        valid_target_sentence_indices = \
            torch.where(special_tokens_mask.view(-1) == 0)[0]
        # Random samples sentences in the batch.
        p = torch.ones(
            len(valid_target_sentence_indices)) / len(valid_target_sentence_indices)
        num_samples = indices_random.sum().tolist()
        if num_samples > 0:
            random_target_sentence_indices = torch.multinomial(p, num_samples=num_samples)
            target_sentence_indices[indices_random] = random_target_sentence_indices
        
        target_sentence_indices.masked_fill_(~masked_indices, -100)
        masked_indices = masked_indices & (indices_replaced | indices_random)
        return target_sentence_indices, masked_indices