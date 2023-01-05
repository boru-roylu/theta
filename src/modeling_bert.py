from typing import Optional, Tuple, Union

from scipy.fftpack import shift

import einops
import torch
import transformers as tfs
from transformers.utils import logging

import modeling_outputs

logger = logging.get_logger(__name__)


class HierarchicalBertModel(tfs.models.bert.modeling_bert.BertPreTrainedModel):
    def __init__(self, config, coordinator_config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = tfs.BertModel(config)

        self.coordinator = tfs.models.bert.modeling_bert.BertEncoder(
            coordinator_config)

        self.cls_token_emb = torch.nn.Parameter(
            torch.randn(1, 1, coordinator_config.hidden_size))

        self.coordinator_type_embeddings = torch.nn.Embedding(
            3, coordinator_config.hidden_size)

        if coordinator_config.hidden_size != config.hidden_size:
            self.proj_for_coordinator = torch.nn.Linear(
                config.hidden_size, coordinator_config.hidden_size)
        if coordinator_config.add_ctx_pooled_output_to_tokens:
            self.output_proj_for_coordinator = torch.nn.Linear(
                coordinator_config.hidden_size, config.hidden_size)

        # Explicit add one more setting becasuse we might use absolute and 
        # relative positional embeddings at the same time.
        if (coordinator_config.add_absolute_position_embeddings
            and coordinator_config.max_position_embeddings > 0):
            self.turn_position_embeddings = torch.nn.Embedding(
                coordinator_config.max_position_embeddings,
                coordinator_config.hidden_size)
            self.register_buffer(
                'turn_position_ids',
                torch.arange(config.max_position_embeddings).expand((1, -1)))

        if coordinator_config.add_ctx_pooled_output_to_tokens:
            self.ctx_LayerNorm = torch.nn.LayerNorm(
                config.hidden_size,
                eps=coordinator_config.layer_norm_eps)
            self.ctx_dropout = torch.nn.Dropout(coordinator_config.hidden_dropout_prob)
        self.coordinator_config = coordinator_config

        # Initialize weights and apply final processing
        self.init_weights()

    def get_input_embeddings(self):
        return self.bert.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.bert.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], modeling_outputs.Seq2SeqLMOutput]:
        if input_ids != None:
            assert input_ids.ndim == 3
            batch_size, num_turns, num_tokens = input_ids.size()
            input_ids = input_ids.view(-1, input_ids.size(-1))
        if attention_mask != None:
            coordinator_attention_mask = torch.clamp(attention_mask.sum(2), max=1)
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        device = input_ids.device

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        bert_pooled_output = encoder_outputs[1]

        # Sentence-level encoding.
        pooled_output = einops.rearrange(
            bert_pooled_output, '(bs nt) hs-> bs nt hs', bs=batch_size)

        if hasattr(self, 'proj_for_coordinator'):
            pooled_output = self.proj_for_coordinator(pooled_output)
        cls_token_embs = einops.repeat(
            self.cls_token_emb, '() nt hs -> bs nt hs', bs=batch_size)
        cls_attention_mask = torch.ones(
            (batch_size, 1), dtype=attention_mask.dtype).to(device)
        coordinator_attention_mask = torch.cat(
            (cls_attention_mask, coordinator_attention_mask), dim=1)

        # num_turns + 1: +1 is for the utterance-level CLS.
        coordinator_type_ids = torch.zeros(
            (batch_size, num_turns+1), dtype=input_ids.dtype).to(device)
        coordinator_type_embs = self.coordinator_type_embeddings(
            coordinator_type_ids)

        pooled_output = torch.cat(
            (cls_token_embs, pooled_output), dim=1)
        pooled_output = pooled_output + coordinator_type_embs

        if hasattr(self, 'turn_position_ids'):
            turn_position_ids = self.turn_position_ids[:, :num_turns+1]
            turn_position_embeddings = \
                self.turn_position_embeddings(turn_position_ids)
            pooled_output = pooled_output + turn_position_embeddings

        extended_coordinator_attention_mask = \
            self.get_extended_attention_mask(
                coordinator_attention_mask,
                pooled_output.size()[:2], device)
                #coordinator_inputs_embs.size()[:2], device)
        coordinator_outputs = self.coordinator(
            #inputs_embeds=pooled_output,
            #attention_mask=coordinator_attention_mask)
            pooled_output,
            attention_mask=extended_coordinator_attention_mask)
        ctx_pooled_output = coordinator_outputs[0]

        # The first index is for the utterance-level CLS token.
        conversation_pooled_output = ctx_pooled_output[:, 0, :]
        ctx_pooled_output = ctx_pooled_output[:, 1:, :]
        coordinator_attention_mask = coordinator_attention_mask[:, 1:]
        ctx_pooled_output = einops.rearrange(
            ctx_pooled_output, 'bs nt hs -> (bs nt) 1 hs')
        coordinator_attention_mask = einops.rearrange(
            coordinator_attention_mask, 'bs nt -> (bs nt) 1')

        if self.coordinator_config.add_ctx_pooled_output_to_tokens:
            sequence_output = sequence_output + \
                self.output_proj_for_coordinator(ctx_pooled_output)
            sequence_output = self.ctx_LayerNorm(sequence_output)
            sequence_output = self.ctx_dropout(sequence_output)

        if not return_dict:
            return (
                sequence_output,
                bert_pooled_output,
                pooled_output,
                ctx_pooled_output) + encoder_outputs[1:]

        # Ignores the utterance-level CLS token.
        pooled_output = pooled_output[:, 1:, :]
        pooled_output = einops.rearrange(pooled_output, 'bs nt hs -> (bs nt) hs')

        return modeling_outputs.HierarchicalBertModelOutput(
            last_hidden_state=sequence_output,
            bert_pooler_output=bert_pooled_output,
            pooler_output=pooled_output,
            ctx_pooler_output=ctx_pooled_output,
            ctx_attentions=coordinator_attention_mask,
            conversation_pooler_output=conversation_pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )