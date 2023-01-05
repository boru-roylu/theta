from typing import Optional, Tuple
import dataclasses
import torch
import transformers as tfs


@dataclasses.dataclass
class Seq2SeqLMOutput(tfs.modeling_outputs.ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    decoder_loss: Optional[torch.FloatTensor] = None
    mlm_logits: Optional[torch.FloatTensor] = None
    mlm_loss: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclasses.dataclass
class HierarchicalBertModelOutput(tfs.modeling_outputs.ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    bert_pooler_output: Optional[torch.FloatTensor] = None
    pooler_output: Optional[torch.FloatTensor] = None
    ctx_pooler_output: Optional[torch.FloatTensor] = None
    ctx_attentions: Optional[torch.FloatTensor] = None
    conversation_pooler_output: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None