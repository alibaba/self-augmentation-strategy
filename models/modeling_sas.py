# coding=utf-8
# Copyright 2020-present the Alibaba Group Holding Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch SAS model. """

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN, get_activation
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    SequenceSummary,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from utils.configuration_sas import SasConfig

from functools import partial

from utils.sas_utils import SequenceSideInfo
from .trainer_callback_sas import SasTrainerCallback, SasWandbCallback, SasTensorBoardCallback
import torch.nn.functional as F
import copy
logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "SasConfig"
_TOKENIZER_FOR_DOC = "SasTokenizer"

SAS_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/sas-small-generator",
    "google/sas-base-generator",
    "google/sas-large-generator",
    "google/sas-small-discriminator",
    "google/sas-base-discriminator",
    "google/sas-large-discriminator",
    # See all SAS models at https://huggingface.co/models?filter=sas
]

class SasEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", ["absolute"])

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.forward
    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0, side_info_sets=dict(),
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if "absolute" in self.position_embedding_type:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->Sas
class SasSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", ["absolute"])

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        side_info_sets=dict(),
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        attention_scores_terms = 1

        attention_scores = attention_scores / math.sqrt(attention_scores_terms)


        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in SasModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class SasSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Sas
class SasAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = SasSelfAttention(config)
        self.output = SasSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        side_info_sets=dict(),
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            side_info_sets,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class SasIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput
class SasOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->Sas
class SasLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = SasAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = SasAttention(config)
        self.intermediate = SasIntermediate(config)
        self.output = SasOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        side_info_sets=dict(),
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            side_info_sets=side_info_sets,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->Sas
class SasEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([SasLayer(config) for _ in range(config.num_hidden_layers)])

        self.position_embedding_type = getattr(config, "position_embedding_type", ["absolute"])
        if "absolute_self_only" in self.position_embedding_type:
            # To be used/shared in all self-attention layers. Copy their dimensions here to be consistent.
            self.self_attention = self.layer[0].attention.self

            self.num_attention_heads = self.self_attention.num_attention_heads
            self.attention_head_size = self.self_attention.attention_head_size
            self.all_head_size = self.self_attention.all_head_size

            self.pos_query = nn.Linear(self.self_attention.query.in_features, self.self_attention.query.out_features)
            self.pos_key = nn.Linear(self.self_attention.key.in_features, self.self_attention.key.out_features)

    def get_position_attention_score(self, hidden_states):
        query_layer = self.self_attention.transpose_for_scores(self.pos_query(hidden_states))
        key_layer = self.self_attention.transpose_for_scores(self.pos_key(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        return attention_scores


    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        side_info_sets=dict(),
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    side_info_sets,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    side_info_sets,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class SasDiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_prediction = nn.Linear(config.hidden_size, 1)
        self.config = config

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = get_activation(self.config.hidden_act)(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze(-1)

        return logits


class SasGeneratorPredictions(nn.Module):
    """Prediction module for the generator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()

        self.LayerNorm = nn.LayerNorm(config.embedding_size)
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)

    def forward(self, generator_hidden_states):
        hidden_states = self.dense(generator_hidden_states)
        hidden_states = get_activation("gelu")(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


class SasPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SasConfig
    base_model_prefix = "sas"
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    _keys_to_ignore_on_load_unexpected = [r"sas\.embeddings_project\.weight", r"sas\.embeddings_project\.bias"]

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


@dataclass
class SasForRTDOutput(ModelOutput):
    """
    Output type of :class:`~transformers.SasForRTD`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss of the SAS objective.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Prediction scores of the head (scores for each token before SoftMax).
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
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class SasForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.SasForPreTraining`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss of the SAS objective.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Prediction scores of the head (scores for each token before SoftMax).
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
    mlm_loss: Optional[torch.FloatTensor] = None
    rtd_loss: Optional[torch.FloatTensor] = None
    mlm_logits: torch.FloatTensor = None
    rtd_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


SAS_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.SasConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

SAS_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.SasTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Sas Model transformer outputting raw hidden-states without any specific head on top. Identical to "
    "the BERT model except that it uses an additional linear layer between the embedding layer and the encoder if the "
    "hidden size and embedding size are different."
    ""
    "Both the generator and discriminator checkpoints may be loaded into this model.",
    SAS_START_DOCSTRING,
)
class SasModel(SasPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = SasEmbeddings(config)

        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)

        self.encoder = SasEncoder(config)
        self.config = config
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)


    @add_start_docstrings_to_model_forward(SAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="google/sas-small-discriminator",
        output_type=BaseModelOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        side_info_sets=dict(),
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        hidden_states = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, side_info_sets=side_info_sets,
        )

        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project(hidden_states)

        hidden_states = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            side_info_sets=side_info_sets,
            return_dict=return_dict,
        )

        return hidden_states


class SasClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        # configure the cls dropout rate
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = nn.Dropout(drop_out)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Sas authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
    
    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def __call__(self, params_ema, params):
        if params_ema is None:
            return params
        return params_ema * self.beta + (1 - self.beta) * params


@add_start_docstrings(
    """
    SAS Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    SAS_START_DOCSTRING,
)
class SasForSequenceClassification(SasPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.sas = SasModel(config)
        self.classifier = SasClassificationHead(config)

        self.mixup_ratio = getattr(config, "mixup_ratio", 0)
        assert(0 <= self.mixup_ratio < 1)

        self.trainer_callback = SasTrainerCallback()
        self.tensorboard_callback = SasTensorBoardCallback()
        self.global_step = None
        self.max_steps = None

        if getattr(config, "contrast_temp", None):
            self.sim = Similarity(temp=config.contrast_temp)

        self.init_weights()

        momentum_encoder_beta = getattr(config, "momentum_encoder_beta", 0)
        self.momentum_encoder_beta = momentum_encoder_beta

        if self.momentum_encoder_beta > 0.0:
            self.params_ema_updater = EMA(self.momentum_encoder_beta)
            self.sas_ema = None
            self.classifier_ema = None

    def get_model_ema(self):
        if self.sas_ema is None:
            self.sas_ema = copy.deepcopy(self.sas)
        else:
            for params_ema, params in zip(self.sas_ema.parameters(), self.sas.parameters()):
                params_ema.data = self.params_ema_updater(params_ema, params)
        if self.classifier_ema is None:
            self.classifier_ema = copy.deepcopy(self.classifier)
        else:
            for params_ema, params in zip(self.classifier_ema.parameters(), self.classifier.parameters()):
                params_ema.data = self.params_ema_updater(params_ema, params)

    def batch_initial_status_update(self):
        if self.training:
            self.global_step = self.trainer_callback.state.global_step
            if self.global_step <= 1:
                self.max_steps = self.trainer_callback.state.max_steps

        mixup_ratio = self.mixup_ratio
        if self.mixup_ratio >= 0.5 and self.global_step < 0.5 * self.max_steps:
            mixup_ratio = 0
        return mixup_ratio

    @add_start_docstrings_to_model_forward(SAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="google/sas-small-discriminator",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        side_info_sets=dict(),
        return_dict=None,
        # Temporary solution: Need the following three fields as placeholder so that they are passed to dataloader (in trainer.py) and then is used for efficient calculation side information embedding in GLUE fine-tuning
        idx=None, 
        ss_sentence_position_in_sequence=None,
        ss_token_position_in_sentence=None,
        # reversed sentence pair input ids, processed by process function in run_glue.py
        reverse_input_ids=None,
        reverse_token_type_ids=None,
        reverse_attention_mask=None,
        # For switch case 
        input_sent1=None,
        input_sent2=None,
        switch_input_ids=None,
        switch_token_type_ids=None,
        switch_attention_mask=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        mixup_ratio = self.batch_initial_status_update()

        self.config.rdrop_weight = getattr(self.config, "rdrop_weight", 0)
        self.config.contrast_weight = getattr(self.config, "contrast_weight", 0)
        self.config.switch_case_prob = getattr(self.config, "switch_case_prob", 0)

        if self.config.rdrop_weight > 0 or self.config.contrast_weight > 0:
            if self.config.switch_case_prob > 0.0:
                input_ids = torch.cat([input_ids, switch_input_ids], dim=0)
                attention_mask = torch.cat([attention_mask, switch_attention_mask], dim=0)
                token_type_ids = torch.cat([token_type_ids, switch_token_type_ids], dim=0)
                if labels is not None:
                    labels = torch.cat([labels, labels], dim=0)
                loss, logits, outputs = self.sas_kl_within_batch(input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
            
            elif self.momentum_encoder_beta > 0.0:
                loss, logits, outputs = self.sas_momentum_kl(input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
            
            else:
                loss, logits, outputs = self.sas_kl(input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
        

        elif self.config.switch_case_prob > 0.0:
            if self.training:
                input_ids = switch_input_ids
                attention_mask = switch_attention_mask
                token_type_ids = switch_token_type_ids
            loss, logits, outputs = self.sas_cross_entropy(input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
        
        elif reverse_input_ids is not None:
            loss, logits, outputs = self.sas_cross_entropy_with_reverse_input(input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
        
        else:
            discriminator_hidden_states = self.sas(
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                head_mask,
                inputs_embeds,
                output_attentions,
                output_hidden_states,
                side_info_sets,
                return_dict,
            )

            sequence_output = discriminator_hidden_states[0]

            logits = self.classifier(sequence_output)
            loss = None
            if labels is not None:
                if not self.training or mixup_ratio == 0 or self.num_labels == 1:
                    if self.num_labels == 1:
                        #  We are doing regression
                        loss_fct = MSELoss()
                        loss = loss_fct(logits.view(-1), labels.view(-1))
                    else:
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                else:  # Do mixup
                    batch_size = sequence_output.shape[0]
                    mixup_idx = np.roll(np.arange(batch_size), shift=-1)
                    sequence_output_mixup = (1 - mixup_ratio) * sequence_output + mixup_ratio * sequence_output[mixup_idx, ...]
                    logits_mixup = self.classifier(sequence_output_mixup)

                    one_hot = torch.zeros_like(logits_mixup).scatter_(-1, labels.view(-1,1), 1)
                    labels_mixup = (1 - mixup_ratio) * one_hot + mixup_ratio * one_hot[mixup_idx, ...]

                    log_prb = F.log_softmax(logits_mixup, dim=1)
                    loss = -(labels_mixup * log_prb).sum(dim=1).mean()

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def sas_cross_entropy_with_reverse_input(self, input_ids, attention_mask,token_type_ids, position_ids, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict,reverse_input_ids,reverse_token_type_ids,reverse_attention_mask):
        prediction_scores_list = []
        outputs_list = []
        for i in range(2):
            if i == 1:
                outputs = self.sas(
                    reverse_input_ids,
                    token_type_ids=reverse_token_type_ids,
                    attention_mask=reverse_attention_mask,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict
                )
            else:
                outputs = self.sas(
                    input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict
                )
            
            sequence_output = outputs[0]
            logits = self.classifier(sequence_output)
            prediction_scores_list.append(logits)
            outputs_list.append(outputs)

        loss = None
        for logits in prediction_scores_list:
            if labels is not None:
                if self.num_labels == 1:
                    loss_fn = torch.nn.MSELoss()
                    if loss:
                        loss += loss_fn(logits.view(-1), labels.view(-1))
                    else:
                        loss = loss_fn(logits.view(-1), labels.view(-1))
                else:
                    loss_fn = CrossEntropyLoss()
                    if loss:
                        loss += loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
                    else:
                        loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        
        if loss is not None:
            if self.num_labels == 1:
                loss_fn = MSELoss()
                rdrop_loss = self.config.reverse_weight * loss_fn(prediction_scores_list[0].view(-1), prediction_scores_list[-1].view(-1))
                loss += rdrop_loss
            else:
                # use kl divergence loss
                p = torch.log_softmax(prediction_scores_list[0].view(-1, self.num_labels), dim=-1)
                p_tec = torch.softmax(prediction_scores_list[0].view(-1, self.num_labels), dim=-1)
                q = torch.log_softmax(prediction_scores_list[-1].view(-1, self.num_labels), dim=-1)
                q_tec = torch.softmax(prediction_scores_list[-1].view(-1, self.num_labels), dim=-1)

                kl_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none').sum()
                reverse_kl_loss = torch.nn.functional.kl_div(q, p_tec, reduction='none').sum()

                rdrop_loss = self.config.reverse_weight * (kl_loss + reverse_kl_loss) / 2.0
                loss += rdrop_loss
        
        return loss, prediction_scores_list[0], outputs_list[0]

    
    def sas_cross_entropy(self, input_ids, attention_mask,token_type_ids, position_ids, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict):
        outputs = self.sas(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fn = torch.nn.MSELoss()
                loss = loss_fn(logits.view(-1), labels.view(-1))
            else:
                loss_fn = CrossEntropyLoss()
                loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits, outputs


    def sas_kl_within_batch(self, input_ids, attention_mask, token_type_ids,
    position_ids, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict):
        batch_size = input_ids.size(0)
        outputs = self.sas(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fn = torch.nn.MSELoss()
                loss = loss_fn(logits.view(-1), labels.view(-1))
            else:
                loss_fn = CrossEntropyLoss()
                loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            loss = 2 * loss  # to match the scale in rdrop

        logits_0, logits_1 = logits[: batch_size // 2], logits[batch_size // 2 :]
        if loss is not None:
            if self.num_labels == 1:
                loss_fn = MSELoss()
                rdrop_loss = self.config.rdrop_weight * loss_fn(logits_0.view(-1), logits_1.view(-1))
                loss += rdrop_loss
            else:
                # use kl divergence loss
                p = torch.log_softmax(logits_0.view(-1, self.num_labels), dim=-1)
                p_tec = torch.softmax(logits_0.view(-1, self.num_labels), dim=-1)
                q = torch.log_softmax(logits_1.view(-1, self.num_labels), dim=-1)
                q_tec = torch.softmax(logits_1.view(-1, self.num_labels), dim=-1)

                kl_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none').sum()
                reverse_kl_loss = torch.nn.functional.kl_div(q, p_tec, reduction='none').sum()

                rdrop_loss = self.config.rdrop_weight * (kl_loss + reverse_kl_loss) / 2.0
                loss += rdrop_loss                       

        # outputs may cause size mis-match
        return loss, logits_0, outputs

    def sas_kl(self, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds, labels, output_attentions, output_hidden_states, 
        return_dict):
        prediction_scores_list = []
        outputs_list = []
        cls_representation_embeddings_list = []
        for i in range(2):
            outputs = self.sas(
                    input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict
                )
            sequence_output = outputs[0]
            if self.config.contrast_weight > 0.0:
                cls_representation_embeddings_list.append(sequence_output[:, 0, :])
            
            if self.config.rdrop_weight > 0.0 or (self.config.rdrop_weight == 0 and i == 0):
                logits = self.classifier(sequence_output)
                prediction_scores_list.append(logits)
                outputs_list.append(outputs)
        
        loss = None
        for logits in prediction_scores_list:
            if labels is not None:
                if self.num_labels == 1:
                    loss_fn = torch.nn.MSELoss()
                    if loss:
                        loss += loss_fn(logits.view(-1), labels.view(-1))
                    else:
                        loss = loss_fn(logits.view(-1), labels.view(-1))
                else:
                    loss_fn = CrossEntropyLoss()
                    if loss:
                        loss += loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
                    else:
                        loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        
        if loss is not None:
            cls_loss = loss.item()
        if loss is not None and self.config.rdrop_weight > 0:
            if self.num_labels == 1:
                loss_fn = MSELoss()
                rdrop_loss = self.config.rdrop_weight * loss_fn(prediction_scores_list[0].view(-1), prediction_scores_list[-1].view(-1))
                loss += rdrop_loss
            
            else:
                # use kl divergence loss
                p = torch.log_softmax(prediction_scores_list[0].view(-1, self.num_labels), dim=-1)
                p_tec = torch.softmax(prediction_scores_list[0].view(-1, self.num_labels), dim=-1)
                q = torch.log_softmax(prediction_scores_list[-1].view(-1, self.num_labels), dim=-1)
                q_tec = torch.softmax(prediction_scores_list[-1].view(-1, self.num_labels), dim=-1)

                if self.config.filter_outliers:
                    kl_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none').sum(1)
                    reverse_kl_loss = torch.nn.functional.kl_div(q, p_tec, reduction='none').sum(1)

                    if torch.distributed.is_initialized() and self.training:
                        kl_loss_list = [torch.zeros_like(kl_loss) for _ in range(torch.distributed.get_world_size())]
                        reverse_kl_loss_list = [torch.zeros_like(reverse_kl_loss) for _ in range(torch.distributed.get_world_size())]
                        torch.distributed.all_gather(tensor_list=kl_loss_list,tensor=kl_loss.contiguous())
                        torch.distributed.all_gather(tensor_list=reverse_kl_loss_list,tensor=reverse_kl_loss.contiguous())
                        kl_loss_list[torch.distributed.get_rank()] = kl_loss
                        reverse_kl_loss_list[torch.distributed.get_rank()] = reverse_kl_loss

                        kl_loss = torch.cat(kl_loss_list, 0)
                        reverse_kl_loss = torch.cat(reverse_kl_loss_list, 0)
                    
                    # filter outliers
                    kl_range = np.quantile(kl_loss.detach().cpu().numpy(), [0.25, 0.75])
                    rv_kl_range = np.quantile(reverse_kl_loss.detach().cpu().numpy(), [0.25, 0.75])

                    kl_loss = torch.where(kl_loss > kl_range[0]-1.5*(kl_range[1]-kl_range[0]), kl_loss, torch.tensor([0.0], device=loss.device))
                    false_positive_count = (kl_loss == 0).sum().item()
                    kl_loss = kl_loss.sum()

                    reverse_kl_loss = torch.where(reverse_kl_loss > rv_kl_range[0] - 1.5*(rv_kl_range[1]-rv_kl_range[0]), reverse_kl_loss, torch.tensor([0.0], device=loss.device))
                    false_positive_count_reverse = (reverse_kl_loss == 0).sum().item()
                    reverse_kl_loss = reverse_kl_loss.sum()
                    
                else:
                    kl_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none').sum()
                    reverse_kl_loss = torch.nn.functional.kl_div(q, p_tec, reduction='none').sum()

                rdrop_loss = self.config.rdrop_weight * (kl_loss + reverse_kl_loss) / 2.0
                loss += rdrop_loss
        
        if loss is not None and self.config.contrast_weight > 0:
            z1, z2 = cls_representation_embeddings_list[0],cls_representation_embeddings_list[1]
            if self.config.sub_dim is not None:
                z1, z2 = z1[:, :self.config.sub_dim], z2[:, :self.config.sub_dim]
            if torch.distributed.is_initialized() and self.training:
                z1_list = [torch.zeros_like(z1) for _ in range(torch.distributed.get_world_size())]
                z2_list = [torch.zeros_like(z2) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(tensor_list=z1_list,tensor=z1.contiguous())
                torch.distributed.all_gather(tensor_list=z2_list,tensor=z2.contiguous())

                z1_list[torch.distributed.get_rank()] = z1
                z2_list[torch.distributed.get_rank()] = z2

                z1 = torch.cat(z1_list, 0)
                z2 = torch.cat(z2_list, 0)

            cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))

            labels = torch.arange(cos_sim.size(0)).long().to(self.device)

            contrast_loss = self.config.contrast_weight * nn.CrossEntropyLoss()(cos_sim, labels)

            loss += contrast_loss
        
        if self.training:
            global_step = self.trainer_callback.state.global_step
            if global_step % 5 == 0:
                print("\r step %d: total loss %.5f classification loss %.5f kl loss %.5f contrast loss %.5f" % (
                    global_step,
                    loss.item(),
                    cls_loss,
                    rdrop_loss.item() if self.config.rdrop_weight > 0 else 0,
                    contrast_loss.item() if self.config.contrast_weight > 0 else 0
                ))
                self.inform['global_step'] = global_step
                self.inform['total_loss'] = loss.item()
                self.inform['classfication_loss'] = cls_loss
                self.inform['rdrop_loss'] = rdrop_loss.item() if self.config.rdrop_weight > 0 else 0
                self.inform['contrast_loss'] = contrast_loss.item() if self.config.contrast_weight > 0 else 0
                self.inform['false_positive_count'] = false_positive_count if self.config.filter_outliers > 0 else 0
                self.inform['false_positive_count_reverse'] = false_positive_count_reverse if self.config.filter_outliers > 0 else 0
                self.tensorboard_callback.log = self.inform
        return loss, prediction_scores_list[0], outputs_list[0]
        

    def sas_momentum_kl(self, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds, labels, output_attentions,output_hidden_states,
        return_dict):
        prediction_scores_list = []
        outputs_list = []
        outputs = self.sas(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        
        prediction_scores_list.append(logits)
        outputs_list.append(outputs)

        with torch.no_grad():
            self.get_model_ema()
            outputs = self.sas_ema(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
            sequence_output = outputs[0]
            logits = self.classifier_ema(sequence_output)
            prediction_scores_list.append(logits)
            outputs_list.append(outputs)
        
        loss = None
        for logits in prediction_scores_list:
            if labels is not None:
                if self.num_labels == 1:
                    loss_fn = torch.nn.MSELoss()
                    if loss:
                        loss += loss_fn(logits.view(-1), labels.view(-1))
                    else:
                        loss = loss_fn(logits.view(-1), labels.view(-1))
                else:
                    loss_fn = CrossEntropyLoss()
                    if loss:
                        loss += loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
                    else:
                        loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        
        if loss is not None:
            cls_loss = loss.item()
        if loss is not None and self.config.rdrop_weight > 0:
            if self.num_labels == 1:
                loss_fn = MSELoss()
                rdrop_loss = self.config.rdrop_weight * loss_fn(prediction_scores_list[0].view(-1), prediction_scores_list[-1].view(-1))
                loss += rdrop_loss
            
            else:
                # use kl divergence loss
                p = torch.log_softmax(prediction_scores_list[0].view(-1, self.num_labels), dim=-1)
                p_tec = torch.softmax(prediction_scores_list[0].view(-1, self.num_labels), dim=-1)
                q = torch.log_softmax(prediction_scores_list[-1].view(-1, self.num_labels), dim=-1)
                q_tec = torch.softmax(prediction_scores_list[-1].view(-1, self.num_labels), dim=-1)

                kl_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none').sum()
                reverse_kl_loss = torch.nn.functional.kl_div(q, p_tec, reduction='none').sum()

                rdrop_loss = self.config.rdrop_weight * (kl_loss + reverse_kl_loss) / 2.0
                loss += rdrop_loss
        
        if self.training:
            global_step = self.trainer_callback.state.global_step
            if global_step % 10 == 0:
                print("\r step %d: total loss %.5f classification loss %.5f kl loss %.5f" % (
                    global_step,
                    loss.item(),
                    cls_loss,
                    rdrop_loss.item() if self.config.rdrop_weight > 0 else 0,
                ))
        return loss, prediction_scores_list[0], outputs_list[0]





@add_start_docstrings(
    """
    Sas model with a binary classification head on top as used during pretraining for identifying generated tokens.

    It is recommended to load the discriminator checkpoint into that model.
    """,
    SAS_START_DOCSTRING,
)
class SasForRTD(SasPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.sas = SasModel(config)
        self.discriminator_predictions = SasDiscriminatorPredictions(config)
        self.init_weights()

    @add_start_docstrings_to_model_forward(SAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SasForRTDOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        side_info_sets=dict(),
        return_dict=None,
    ):
        r"""
        labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the SAS loss. Input should be a sequence of tokens (see :obj:`input_ids`
            docstring) Indices should be in ``[0, 1]``:

            - 0 indicates the token is an original token,
            - 1 indicates the token was replaced.

        Returns:

        Examples::

            >>> from transformers import SasTokenizer, SasForRTD
            >>> import torch

            >>> tokenizer = SasTokenizer.from_pretrained('google/sas-small-discriminator')
            >>> model = SasForRTD.from_pretrained('google/sas-small-discriminator')

            >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            >>> logits = model(input_ids).logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.sas(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            side_info_sets,
            return_dict,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]

        logits = self.discriminator_predictions(discriminator_sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1, discriminator_sequence_output.shape[1]) == 1
                active_logits = logits.view(-1, discriminator_sequence_output.shape[1])[active_loss]
                active_labels = labels[active_loss]
                loss = loss_fct(active_logits, active_labels.float())
            else:
                loss = loss_fct(logits.view(-1, discriminator_sequence_output.shape[1]), labels.float())

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return SasForRTDOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )

@add_start_docstrings(
    """
    Sas model with a binary classification head on top as used during pretraining for identifying generated tokens.

    It is recommended to load the discriminator checkpoint into that model.
    """,
    SAS_START_DOCSTRING,
)
class SasForPreTraining(SasPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.sas = SasModel(config)
        self.generator_predictions = SasGeneratorPredictions(config)
        self.generator_lm_head = nn.Linear(config.embedding_size, config.vocab_size)

        self.discriminator_predictions = SasDiscriminatorPredictions(config)
        self.init_weights()

        self.config = config
        self.global_step = 0
        self.max_steps = None
        self.epoch = 0
        self.max_epochs_int = None
        self.dataset = None
        self.metrics = dict()
        self.inform = dict()

        # Saved for self-augmentation
        self.mlm_positions = None
        self.mlm_labels = None
        self.mlm_logits = None

        self.gen_weight = 1
        self.dis_weight = 1

        self.flag_debugExtraMetrics = False
        self.time = time.time()
        self.durations = dict()
        self.max_memory_allocated = 0
        self.memory_allocated = 0


        self.trainer_callback = SasTrainerCallback()
        self.wandb_callback = SasWandbCallback()
        self.tensorboard_callback = SasTensorBoardCallback()

    def get_output_embeddings(self):
        return self.generator_lm_head

    def set_output_embeddings(self, word_embeddings):
        self.generator_lm_head = word_embeddings


    # Purpose: Track hardware/compute (cpu/memory & cuda memory usage, etc.). Normalizer is used to show normalized usage (e.g., MLM's CUDA usage per MLM head)
    def track_compute_statistics(self, mod, inp, out, name: Optional[str] = 'Unknown'):
        if not self.training:
            return 

        current_time = time.time()
        # Note: use comma "," to seperate metrics during printing, in order to copy-paste log to Excel for easier analysis
        # In Excel, try "Data, then Text to Columns" to seperate fields by comma

        if torch.cuda.is_available() and \
            ((self.global_step + 1) % self.config.debug_config['debugMemStatsInterval'] in [0, 1] or self.global_step == 10):

            self.durations[name] = (current_time - self.time)
            self.inform["gpu/max_memory_reserved_%s" % (name)] = torch.tensor(torch.cuda.max_memory_reserved()).float().item() / 1e6
            self.inform["gpu/max_memory_cached_%s" % (name)] = torch.tensor(torch.cuda.max_memory_cached()).float().item() / 1e6
            self.inform["gpu/max_memory_allocated_%s" % (name)] = torch.tensor(torch.cuda.max_memory_allocated()).float().item() / 1e6
            self.inform["gpu/memory_allocated_%s" % (name)] = torch.tensor(torch.cuda.memory_allocated()).float().item() / 1e6
            torch.cuda.reset_peak_memory_stats()

            printStr = 'CUDA Allocated (M), %04d, %04d, %04d, Change, %03.2f, %03.2f' % (
                self.memory_allocated,
                self.inform["gpu/max_memory_allocated_%s" % (name)],
                self.inform["gpu/memory_allocated_%s" % (name)],
                self.inform["gpu/max_memory_allocated_%s" % (name)] - self.memory_allocated,
                self.inform["gpu/memory_allocated_%s" % (name)] - self.inform["gpu/max_memory_allocated_%s" % (name)]
            )

            self.max_memory_allocated = self.inform["gpu/max_memory_allocated_%s" % (name)]
            self.memory_allocated = self.inform["gpu/memory_allocated_%s" % (name)]

            new_time = time.time()
            print('%sE%d.%03d, track_compute_statistics, %s, %s, \t@ Took, %.4f, sec, %s' % (
                "\n" if 'Fwd,Pre,Other' in name else "",
                self.epoch, self.global_step, 
                name.ljust(40, ' '), printStr, 
                new_time - current_time, datetime.datetime.now())
            )

            self.time = new_time

        else:
            self.time = current_time

    # Purpose: Track extra algorithm metrics (such as loss, accuracy, etc. that is useful to track and debug algorithm behaviors)
    def track_extra_algorithm_metrics(self, topic: str, **kwargs):

        # Jingqiao's Special Note: This calculation is expensive (both memory and time).
        if not (self.flag_debugExtraMetrics or topic == 'end_of_forward'):
            return

        with torch.no_grad():

            if topic == 'end_of_forward':
                self.metrics['total_loss'] = kwargs['total_loss']

                for metric_name in self.metrics:
                    self.inform['sanity/' + self.tb_prefix + metric_name] = self.metrics[metric_name].item()

                if self.flag_debugExtraMetrics:
                    self.wandb_callback.log = self.inform           # pass it to the trainer to log extra metrics
                    self.tensorboard_callback.log = self.inform     # pass it to the trainer to log extra metrics

                if self.training and self.global_step % self.config.debug_config['logging_steps'] == 0:
                    print("\rE%d.%03d: [Dis+Gen] Loss %.5f + %.5f; Acc: %.5f + %.5f" % (
                        self.epoch, self.global_step,
                        float(self.metrics['dis_loss']) if 'dis_loss' in self.metrics else 0, 
                        float(self.metrics['gen_loss']),
                        float(self.metrics['dis_accuracy'] if 'dis_accuracy' in self.metrics else 0), 
                        float(self.metrics['gen_accuracy'])), end='')

            elif topic == 'mlm':
                self.metrics['gen_loss'] = kwargs['mlm_loss']

                self.inform['sanity/' + self.tb_prefix + 'mask_ratio_2gram'] = torch.logical_and(
                    self.mlm_positions, 
                    torch.roll(self.mlm_positions, shifts=1, dims=1) == self.mlm_positions
                ).sum().item() * 2.0 / self.mlm_positions.numel()
                    
                self.inform['sanity/' + self.tb_prefix + 'mask_ratio_total'] = \
                    self.mlm_positions.sum().item() / self.mlm_positions.numel()

                self.inform['sanity/' + self.tb_prefix + 'mask_2gram_ratio'] = \
                    self.inform['sanity/' + self.tb_prefix + 'mask_ratio_2gram'] / self.inform['sanity/' + self.tb_prefix + 'mask_ratio_total']

                # In case of "alow hit", this calcuation can be done after generating fake_token so that we don't need to self.sm(logits) twice.
                # In case of "not allow hit", however, we have to do it before generating fake_token (otherwise, the prob of the hard label is already set to 0)
                softmax_fct = torch.nn.Softmax(dim=-1)
                probs = softmax_fct(self.mlm_logits)
                fake_top_label_prob, fake_top_label = torch.topk(probs, 10)
                hit_top1 = fake_top_label[..., 0] == self.mlm_labels
                hit_top2to10 = fake_top_label[..., 1:] == self.mlm_labels.unsqueeze(1)

                self.metrics['gen_accuracy'] = hit_top1.float().mean()
                self.metrics['gen_probability'] = fake_top_label_prob[..., 0].mean()
                self.metrics['gen_accuracy_top2to10'] = hit_top2to10.sum(dim=-1).float().mean()
                self.metrics['gen_prob_top2to10'] = fake_top_label_prob[..., 1:10].sum(dim=-1).mean()

            elif topic == 'rtd':
                self.metrics['dis_loss'] = kwargs['rtd_loss']
                rtd_logits, rtd_labels = kwargs['rtd_logits'], kwargs['rtd_labels']
                dis_predictions = torch.round((torch.sign(rtd_logits) + 1) / 2).int()
                neg_positions = (rtd_labels == 0)

                self.metrics['dis_accuracy_base'] = neg_positions.float().mean()
                self.metrics['dis_accuracy']      = ((dis_predictions == rtd_labels) * 1.0).mean()
                self.metrics['dis_accuracy_neg']  = ((dis_predictions[neg_positions] == rtd_labels[neg_positions]) * 1.0).mean()
                self.metrics['dis_accuracy_pos']  = ((dis_predictions[~neg_positions] == rtd_labels[~neg_positions]) * 1.0).mean()

    def batch_initial_status_update(self, side_info_sets):
        self.metrics = dict()
        self.inform = dict()

        self.tb_prefix = '' if self.training else 'E-'
        self.batch_idx = side_info_sets['batch_idx']

        if self.training:
            self.global_step = self.trainer_callback.state.global_step
            self.epoch = self.trainer_callback.state.epoch
            self.inform['all/global_step'] = self.global_step
            self.inform['all/global_epoch'] = self.epoch
            if self.max_steps is None:
                self.max_steps = self.trainer_callback.state.max_steps
                self.max_epochs_int = self.trainer_callback.state.num_train_epochs
                
            self.schedule_loss_weights()

            self.flag_debugExtraMetrics = self.config.debug_config['debugExtraMetrics'] and (
                self.global_step < 100 or self.global_step % self.config.debug_config['debugExtraMetrics'] == 0)
        else:
            self.flag_debugExtraMetrics = True

    def schedule_loss_weights(self):
        if self.global_step == 1:  # Check it once at the beginning of the training
            assert self.config.dis_weight_scheduler in [0, 1, 2, 3, 4], "Don't support other dis_weight_scheduler yet"

        # gen_weight_scheduler
        self.gen_weight = self.config.gen_weight

        # dis_weight_scheduler
        self.dis_weight_setting = np.fromstring(self.config.dis_weight, dtype=float, sep='-')
        if self.config.dis_weight_scheduler == 0:
            self.dis_weight = self.dis_weight_setting[0]
        elif self.config.dis_weight_scheduler in [1, 2]:
            training_perc = min(1, self.global_step / self.max_steps)
            if self.config.dis_weight_scheduler == 1:       # Update every step
                pass
            elif self.config.dis_weight_scheduler == 2:     # Update at the end of each epoch
                training_perc = min(1, int(self.epoch) / (self.max_epochs_int - 1))

            self.dis_weight = self.dis_weight_setting[0] * (1 - training_perc) + self.dis_weight_setting[1] * training_perc

        elif self.config.dis_weight_scheduler == 4:
            if self.epoch < self.config.cold_start_epochs:
                self.dis_weight = self.dis_weight_setting[0]
            else:
                self.dis_weight = self.dis_weight_setting[1]

        self.inform['loss_weights/dis_weight'] = self.dis_weight


        training_perc = min(1, self.global_step / (self.max_steps-1))

    # Purpose: Prepare the data for SA and LS.
    def run_SA_and_LS_preparation(self, inform=dict()):

        if not self.training:
            return inform
        if not (self.dataset and self.batch_idx is not None):
            return inform

        # Don't need gradient information during SA and LS's data preparation
        with torch.no_grad():
            self.inform = {**self.inform, **inform}

            # No longer need to clone, if we do the SA and LA after backward step
            mlm_logits = self.mlm_logits
            mlm_positions = self.mlm_positions
            mlm_labels = self.mlm_labels


            # Freeze generation
            flag_freeze_generation = (self.config.augmentation_copies > 1 and self.epoch >= self.config.cold_start_epochs)
            # Shift the multiple augmentations generated per instance before generation freezing, so that different augmentations are used in different epochs for each instance after freezing.
            if flag_freeze_generation and self.training:
                self.dataset.update_buffer_da(self.batch_idx.cpu().data.numpy(), flag_freeze_generation=flag_freeze_generation)

            # To save computation time, only start generating data augmentation in the last epoch of the cold start period.
            num_to_sample_da = (self.config.augmentation_copies if not flag_freeze_generation and self.epoch >= self.config.cold_start_epochs - 1 
                                else 0)

            if num_to_sample_da > 0 or self.flag_debugExtraMetrics:
                softmax_fct = torch.nn.Softmax(dim=-1)
                if self.config.augmentation_temperature < 0:     # In case of being negative, use no-hit schedule
                    # Avoid modify logits directly, because it might be used in some other place
                    logits_non_label = mlm_logits.clone().detach()
                    logits_non_label.scatter_(-1, mlm_labels.view(-1, 1), -10000)
                    probs = softmax_fct(logits_non_label / abs(self.config.augmentation_temperature))
                else:
                    probs = softmax_fct(mlm_logits / abs(self.config.augmentation_temperature))

                if torch.isnan(probs).any():
                    probs[...] = 1.0 / probs.shape[1]

                # Set num_samples to a minimal 2, otherwise multinomial is not controlled by random seed
                # (a bug in multinomial - happens when some element of probs is less than 1e-5 when using float16)
                # For now, we use this as a temporary workaround to make experiments reproducible.
                fake_token = torch.multinomial(probs, num_samples=max(2, num_to_sample_da), replacement=True)

                self.track_compute_statistics(mod=None, inp=None, out=None, name='DataAugmentation,Multinomial etc.,')

                if num_to_sample_da > 0:
                    self.dataset.save_buffer_da(self.batch_idx.cpu().data.numpy(), fake_token[..., :num_to_sample_da],
                                                aug_positions=mlm_positions.cpu())

                if self.flag_debugExtraMetrics:
                    hit = (fake_token[..., 0] == mlm_labels)
                    fake_prob = torch.gather(probs, 1, fake_token[..., :1])
                    self.inform['sanity/' + self.tb_prefix + 'hit_ratio_mask'] = hit.float().mean().item()
                    self.inform['sanity/' + self.tb_prefix + 'fake_prob_average'] = fake_prob.mean().item()
                    self.inform['sanity/' + self.tb_prefix + 'fake_prob_larger_50%'] = ((fake_prob > 0.5) * 1.0).mean().item()

                    label_probs = torch.gather(probs, dim=1, index=mlm_labels.view(-1, 1)) 
                    self.inform['sanity/' + self.tb_prefix + 'label_probs'] = label_probs.mean().item()

                self.track_compute_statistics(mod=None, inp=None, out=None, name='DataAugmentation,SaveBuffer,')

                if self.flag_debugExtraMetrics:
                    self.wandb_callback.log = self.inform           # pass it to the trainer to log extra metrics
                    self.tensorboard_callback.log = self.inform     # pass it to the trainer to log extra metrics

        return self.inform  # Update inform in case of calculation after backward, so that SA and LS's inform can be logged in the training process


    @add_start_docstrings_to_model_forward(SAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SasForRTDOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        head_mask: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        mlm_labels: torch.Tensor = None,
        rtd_labels: torch.Tensor = None,
        output_attentions=None,
        output_hidden_states=None,
        side_info_sets: Optional[dict] = {},
        return_dict=None,
    ):
        r"""
        labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the SAS loss. Input should be a sequence of tokens (see :obj:`input_ids`
            docstring) Indices should be in ``[0, 1]``:

            - 0 indicates the token is an original token,
            - 1 indicates the token was replaced.

        Returns:

        Examples::

            >>> from transformers import SasTokenizer, SasForPreTraining
            >>> import torch

            >>> tokenizer = SasTokenizer.from_pretrained('google/sas-small-discriminator')
            >>> model = SasForPreTraining.from_pretrained('google/sas-small-discriminator')

            >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            >>> logits = model(input_ids).logits
        """
        self.batch_initial_status_update(side_info_sets)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.sas(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            side_info_sets,
            return_dict,
        )

        # Masked language modeling
        self.generator_sequence_output = hidden_states[0]
        self.mlm_positions = (mlm_labels != -100)
        self.mlm_labels = mlm_labels[self.mlm_positions]
        self.mlm_logits = self.generator_lm_head(self.generator_predictions(self.generator_sequence_output[self.mlm_positions]))

        loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token


        mlm_loss = loss_fct(self.mlm_logits.view(-1, self.config.vocab_size), self.mlm_labels.view(-1))

        self.track_extra_algorithm_metrics(topic='mlm', mlm_loss=mlm_loss)     

        # RTD task
        if rtd_labels is not None and self.dis_weight > 0:
            discriminator_sequence_output = hidden_states[0]
            rtd_logits = self.discriminator_predictions(discriminator_sequence_output)
            loss_fct = nn.BCEWithLogitsLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1, discriminator_sequence_output.shape[1]) == 1
                active_logits = rtd_logits.view(-1, discriminator_sequence_output.shape[1])[active_loss]
                active_labels = rtd_labels[active_loss]
                rtd_loss = loss_fct(active_logits, active_labels.float())
            else:
                rtd_loss = loss_fct(rtd_logits.view(-1, discriminator_sequence_output.shape[1]), rtd_labels.float())

            self.track_extra_algorithm_metrics(topic='rtd', rtd_loss=rtd_loss, rtd_logits=rtd_logits, rtd_labels=rtd_labels)
        else:
            rtd_loss = 0; rtd_logits = None

        loss = self.gen_weight * mlm_loss + self.dis_weight * rtd_loss
        self.track_extra_algorithm_metrics(topic='end_of_forward', total_loss=loss)    

        self.run_SA_and_LS_preparation()

        if not return_dict:
            output = (self.mlm_logits, rtd_logits,) + hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return SasForPreTrainingOutput(
            loss=loss,
            mlm_loss=mlm_loss,
            mlm_logits=self.mlm_logits,
            rtd_loss=rtd_loss,
            rtd_logits=rtd_logits,
            hidden_states=hidden_states.hidden_states,
            attentions=hidden_states.attentions,
        )



@add_start_docstrings(
    """
    Sas model with a language modeling head on top.

    Even though both the discriminator and generator may be loaded into this model, the generator is the only model of
    the two to have been trained for the masked language modeling task.
    """,
    SAS_START_DOCSTRING,
)
class SasForMaskedLM(SasPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.sas = SasModel(config)
        self.generator_predictions = SasGeneratorPredictions(config)

        self.generator_lm_head = nn.Linear(config.embedding_size, config.vocab_size)
        self.init_weights()

    def get_output_embeddings(self):
        return self.generator_lm_head

    def set_output_embeddings(self, word_embeddings):
        self.generator_lm_head = word_embeddings

    @add_start_docstrings_to_model_forward(SAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="google/sas-small-discriminator",
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        side_info_sets=dict(),
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        generator_hidden_states = self.sas(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            side_info_sets,
            return_dict,
        )
        generator_sequence_output = generator_hidden_states[0]

        prediction_scores = self.generator_predictions(generator_sequence_output)
        prediction_scores = self.generator_lm_head(prediction_scores)

        loss = None
        # Masked language modeling softmax layer
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + generator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=generator_hidden_states.hidden_states,
            attentions=generator_hidden_states.attentions,
        )


@add_start_docstrings(
    """
    Sas model with a token classification head on top.

    Both the discriminator and generator may be loaded into this model.
    """,
    SAS_START_DOCSTRING,
)
class SasForTokenClassification(SasPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.sas = SasModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    @add_start_docstrings_to_model_forward(SAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="google/sas-small-discriminator",
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        side_info_sets=dict(),
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.sas(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            side_info_sets,
            return_dict,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]

        discriminator_sequence_output = self.dropout(discriminator_sequence_output)
        logits = self.classifier(discriminator_sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.config.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )


@add_start_docstrings(
    """
    SAS Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    SAS_START_DOCSTRING,
)
class SasForQuestionAnswering(SasPreTrainedModel):
    config_class = SasConfig
    base_model_prefix = "sas"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.sas = SasModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(SAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="google/sas-small-discriminator",
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        side_info_sets=dict(),
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.sas(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            side_info_sets=side_info_sets,
        )

        sequence_output = discriminator_hidden_states[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (
                start_logits,
                end_logits,
            ) + discriminator_hidden_states[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )


@add_start_docstrings(
    """
    SAS Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    SAS_START_DOCSTRING,
)
class SasForMultipleChoice(SasPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.sas = SasModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    @add_start_docstrings_to_model_forward(SAS_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="google/sas-small-discriminator",
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        side_info_sets=dict(),
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        discriminator_hidden_states = self.sas(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            side_info_sets=side_info_sets,
            return_dict=return_dict,
        )

        sequence_output = discriminator_hidden_states[0]

        pooled_output = self.sequence_summary(sequence_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )
