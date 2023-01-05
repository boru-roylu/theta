import dataclasses
from typing import Optional, Tuple

import einops
import torch
import torch.nn as nn
import transformers as tfs


def make_hierarchical(forward):
    def hierarchical_forward(self, *args, **kwargs):
        batch_size, num_turns, max_seq_len = kwargs['input_ids'].size()

        # Flattens input_ids and attention_mask for transformer.
        input_ids = kwargs['input_ids'].view(-1, max_seq_len)
        attention_mask = kwargs['attention_mask'].view(-1, max_seq_len)

        batch = {'input_ids': input_ids, 'attention_mask': attention_mask}
        outputs = forward(self, **batch)
        pooled_output = outputs['pooler_output']
        pooled_output = pooled_output.view(batch_size, num_turns, -1)
        outputs['pooler_output'] = pooled_output

        last_hidden_state = outputs['last_hidden_state']
        last_hidden_state = last_hidden_state.view(
            batch_size, num_turns, max_seq_len, -1)
        attention_mask = kwargs['attention_mask'].unsqueeze(-1)
        numerator = torch.sum(attention_mask * last_hidden_state, dim=2)
        denominator = torch.clip(torch.sum(attention_mask, dim=2), min=1)
        mean_embeddings = numerator / denominator
        outputs['mean_output'] = mean_embeddings
        return outputs
    return hierarchical_forward


@dataclasses.dataclass
class StateClassifierOutput(tfs.file_utils.ModelOutput):
    """
    Base class for outputs of sentence classification models.
    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    coordinator_loss: Optional[torch.FloatTensor] = None
    state_loss: Optional[torch.FloatTensor] = None
    regularization_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class HierarchicalBertModel(tfs.BertModel):
    @make_hierarchical
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class HierarchicalBertModelForConversationClassification(
    tfs.models.bert.BertPreTrainedModel):
    def __init__(self,
                 config,
                 coordinator_config,
                 num_states,
                 num_clusters,
                 use_state_sequence_classifier=False,
                 use_cluster_sequence_classifier=False,
                 state_sequence_encoder_type=None,
                 cluster_sequence_encoder_type=None,
                 num_labels=None):

        super().__init__(config)

        if num_labels:
            self.num_labels = num_labels
        else:
            self.num_labels = config.num_labels
        self.config = config

        self.bert = HierarchicalBertModel(config)

        if config.classifier_dropout is None:
            classifier_dropout = config.hidden_dropout_prob
        else:
            classifier_dropout = config.classifier_dropout 

        if coordinator_config.hidden_size != config.hidden_size:
            self.proj_for_coordinator = nn.Linear(
                config.hidden_size, coordinator_config.hidden_size)

        self.coordinator = tfs.models.bert.modeling_bert.BertEncoder(
            coordinator_config)
        self.cls_token_emb = nn.Parameter(
            torch.randn(1, 1, coordinator_config.hidden_size))
        self.coordinator_position_embeddings = nn.Embedding(
            coordinator_config.max_position_embeddings,
            coordinator_config.hidden_size)

        self.register_buffer(
            "position_ids",
            torch.arange(
                coordinator_config.max_position_embeddings).expand((1, -1)))

        if use_state_sequence_classifier:
            self.state_layer_norm = nn.LayerNorm(
                coordinator_config.hidden_size, eps=config.layer_norm_eps)
            self.state_dropout = nn.Dropout(config.hidden_dropout_prob)
            self.state_embeddings = torch.nn.Embedding(
                num_states, coordinator_config.hidden_size)
            self.state_cls_token_emb = nn.Parameter(
                torch.randn(1, 1, coordinator_config.hidden_size))

            if state_sequence_encoder_type == 'dnn':
                self.state_sequence_encoder = nn.Linear(
                    coordinator_config.hidden_size,
                    coordinator_config.hidden_size)
            elif state_sequence_encoder_type == 'transformer':
                self.state_sequence_encoder = \
                    tfs.models.bert.modeling_bert.BertEncoder(
                        coordinator_config)
            else:
                raise ValueError(f'{state_sequence_encoder_type = } is '
                                 f'not supported.')

        if use_cluster_sequence_classifier:
            self.cluster_layer_norm = nn.LayerNorm(
                coordinator_config.hidden_size, eps=config.layer_norm_eps)
            self.cluster_dropout = nn.Dropout(config.hidden_dropout_prob)
            self.cluster_embeddings = torch.nn.Embedding(
                num_clusters, coordinator_config.hidden_size)
            self.cluster_cls_token_emb = nn.Parameter(
                torch.randn(1, 1, coordinator_config.hidden_size))

            if cluster_sequence_encoder_type == 'dnn':
                self.cluster_sequence_encoder = nn.Linear(
                    coordinator_config.hidden_size,
                    coordinator_config.hidden_size)
            elif cluster_sequence_encoder_type == 'transformer':
                self.cluster_sequence_encoder = \
                    tfs.models.bert.modeling_bert.BertEncoder(
                        coordinator_config)
            else:
                raise ValueError(f'{cluster_sequence_encoder_type = } is '
                                 f'not supported.')

        classifier_hidden_size = coordinator_config.hidden_size
        if use_state_sequence_classifier:
            classifier_hidden_size += coordinator_config.hidden_size
        if use_cluster_sequence_classifier:
            classifier_hidden_size += coordinator_config.hidden_size

        # TODO(roylu): Figures out where change the value num_labels.
        self.classifier = nn.Linear(classifier_hidden_size, config.num_labels)

        # 2 types: text and state parts.
        self.coordinator_type_embeddings = torch.nn.Embedding(
            2, coordinator_config.hidden_size)

        self.dropout = nn.Dropout(classifier_dropout)

        self.class_weight = None
        self.num_states = num_states
        self.num_clusters = num_clusters
        self.use_state_sequence_classifier = use_state_sequence_classifier
        self.use_cluster_sequence_classifier = use_cluster_sequence_classifier
        self.state_sequence_encoder_type = state_sequence_encoder_type
        self.cluster_sequence_encoder_type = cluster_sequence_encoder_type

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        next_sentence_label=None,
        output_attentions=None,
        output_hidden_states=None,
        state_input_ids=None,
        state_attention_mask=None,
        cluster_input_ids=None,
        cluster_attention_mask=None,
        return_dict=None):

        if return_dict is None:
            return_dict = self.config.use_return_dict
        else:
            return_dict = return_dict

        batch_size, num_turns, max_seq_len = input_ids.size()
        #input_ids = input_ids.view(-1, max_seq_len)
        #attention_mask = attention_mask.view(-1, max_seq_len)

        # Uses input_ids as kwargs not args because we take the input_ids in 
        # decorator, make_hierarchical, by kwargs.
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        text_pooled_output = outputs['pooler_output']
        # Projects before feeding into coordinator to make vector size aligned.
        text_pooled_output = self.proj_for_coordinator(text_pooled_output)
        batch_size, text_sequence_len = text_pooled_output.size()[:-1]

        # sequence_len is block-level.
        device = text_pooled_output.device
        dtype = text_pooled_output.dtype 

        # Text part.
        text_attention_mask = torch.clip(attention_mask.sum(dim=2), 0, 1)
        cls_token_embs = einops.repeat(
            self.cls_token_emb, '() n d -> b n d', b=batch_size)
        text_pooled_output = torch.cat(
            (cls_token_embs, text_pooled_output), dim=1)
        cls_text_attention_mask = torch.ones(
            (batch_size, 1), dtype=attention_mask.dtype).to(device)
        text_attention_mask = torch.cat(
            (cls_text_attention_mask, text_attention_mask), dim=1)
        # +1 is for text CLS.
        text_sequence_len += 1
        text_type_ids = torch.zeros((batch_size, text_sequence_len),
                                    dtype=input_ids.dtype).to(device)

        # State view.
        add_state_sequence = state_input_ids is not None
        if add_state_sequence:
            batch_size, state_sequence_len = state_input_ids.size()
            state_embs = self.state_embeddings(state_input_ids)
            state_embs = self.state_layer_norm(state_embs)
            state_embs = self.state_dropout(state_embs)

            # Prepends state_cls_token_emb later, since DNN doesn't require it.
            state_cls_token_embs = einops.repeat(
                self.state_cls_token_emb, '() n d -> b n d', b=batch_size)
            state_embs = torch.cat((state_cls_token_embs, state_embs), dim=1)

            state_cls_attention_mask = torch.ones(
                (batch_size, 1), dtype=attention_mask.dtype).to(device)
            state_attention_mask = torch.cat(
                (state_cls_attention_mask, state_attention_mask), dim=1)
            # +1 is for state CLS.
            state_sequence_len += 1
            state_type_ids = torch.ones(
                (batch_size, state_sequence_len),
                dtype=state_input_ids.dtype).to(device)

            # State classifier.
            if self.use_state_sequence_classifier:
                if self.state_sequence_encoder_type == 'dnn':
                    state_embs = state_embs.mean(dim=1)
                    state_pooled_output = self.state_sequence_encoder(state_embs)
                elif self.state_sequence_encoder_type == 'transformer':
                    state_input_shape = (batch_size, state_sequence_len)
                    extended_state_attention_mask = self.get_extended_attention_mask(
                        state_attention_mask, state_input_shape, device)
                    state_pooled_output = self.state_sequence_encoder(
                        state_embs, attention_mask=extended_state_attention_mask)
                    state_pooled_output = state_pooled_output['last_hidden_state'][:, 0, :]

        # Cluster view.
        add_cluster_sequence = cluster_input_ids is not None
        if add_cluster_sequence:
            batch_size, cluster_sequence_len = cluster_input_ids.size()
            cluster_embs = self.cluster_embeddings(cluster_input_ids)
            cluster_embs = self.cluster_layer_norm(cluster_embs)
            cluster_embs = self.cluster_dropout(cluster_embs)

            # Prepends cluster_cls_token_emb later, since DNN doesn't require it.
            cluster_cls_token_embs = einops.repeat(
                self.cluster_cls_token_emb, '() n d -> b n d', b=batch_size)
            cluster_embs = torch.cat((cluster_cls_token_embs, cluster_embs), dim=1)

            cluster_cls_attention_mask = torch.ones(
                (batch_size, 1), dtype=attention_mask.dtype).to(device)
            cluster_attention_mask = torch.cat(
                (cluster_cls_attention_mask, cluster_attention_mask), dim=1)
            # +1 is for cluster CLS.
            cluster_sequence_len += 1
            cluster_type_ids = torch.ones(
                (batch_size, cluster_sequence_len),
                dtype=cluster_input_ids.dtype).to(device)

            # cluster classifier.
            if self.use_cluster_sequence_classifier:
                if self.cluster_sequence_encoder_type == 'dnn':
                    cluster_embs = cluster_embs.mean(dim=1)
                    cluster_pooled_output = self.cluster_sequence_encoder(cluster_embs)
                elif self.cluster_sequence_encoder_type == 'transformer':
                    cluster_input_shape = (batch_size, cluster_sequence_len)
                    extended_cluster_attention_mask = self.get_extended_attention_mask(
                        cluster_attention_mask, cluster_input_shape, device)
                    cluster_pooled_output = self.cluster_sequence_encoder(
                        cluster_embs, attention_mask=extended_cluster_attention_mask)
                    cluster_pooled_output = cluster_pooled_output['last_hidden_state'][:, 0, :]

        # Coordinator (text or joint classifier).
        coordinator_attention_mask = text_attention_mask
        coordinator_sequence_len = text_sequence_len
        coordinator_input_embs = text_pooled_output
        coordinator_type_ids = text_type_ids

        # Uses one coordinator only: no seperate state sequence classifier.
        if not self.use_state_sequence_classifier and add_state_sequence:
            coordinator_sequence_len += state_sequence_len
            coordinator_attention_mask = torch.cat(
                (text_attention_mask, state_attention_mask), dim=1)
            coordinator_input_embs = torch.cat(
                (text_pooled_output, state_embs), dim=1)
            coordinator_type_ids = torch.cat(
                (text_type_ids, state_type_ids), dim=1)

        if not self.use_cluster_sequence_classifier and add_cluster_sequence:
            coordinator_sequence_len += cluster_sequence_len
            coordinator_attention_mask = torch.cat(
                (text_attention_mask, cluster_attention_mask), dim=1)
            coordinator_input_embs = torch.cat(
                (text_pooled_output, cluster_embs), dim=1)
            coordinator_type_ids = torch.cat(
                (text_type_ids, cluster_type_ids), dim=1)

        coordinator_type_embs = self.coordinator_type_embeddings(
            coordinator_type_ids)
        position_ids = self.position_ids[:, coordinator_sequence_len]
        coordinator_position_embs = self.coordinator_position_embeddings(
            position_ids)

        coordinator_input_shape = (batch_size, coordinator_sequence_len)
        extended_coordinator_attention_mask = self.get_extended_attention_mask(
            coordinator_attention_mask, coordinator_input_shape, device)

        coordinator_input_embs += coordinator_type_embs
        coordinator_input_embs += coordinator_position_embs

        coordinator_pooled_output = self.coordinator(
            coordinator_input_embs,
            attention_mask=extended_coordinator_attention_mask)
        coordinator_pooled_output = coordinator_pooled_output['last_hidden_state'][:, 0, :]
        coordinator_pooled_output = self.dropout(coordinator_pooled_output)

        concat_pooled_outputs = [coordinator_pooled_output] 

        if add_state_sequence:
            concat_pooled_outputs.append(state_pooled_output)
        if add_cluster_sequence:
            concat_pooled_outputs.append(cluster_pooled_output)

        concat_pooled_output = torch.cat(concat_pooled_outputs, dim=1)
        coordinator_logits = self.classifier(concat_pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            
            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(coordinator_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(coordinator_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                if self.class_weight:
                    loss_fct = nn.CrossEntropyLoss(self.class_weight.type(dtype))
                else:
                    loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(coordinator_logits.view(-1, self.num_labels),
                                labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(coordinator_logits, labels)

        return tfs.modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=coordinator_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions)