import os
import logging
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore import Parameter, Tensor
from mindspore.common.initializer import initializer, TruncatedNormal
import mindspore.numpy as mnp
from typing import Optional
from mindspore.common.initializer import initializer, Normal, Uniform, HeUniform, _calculate_fan_in_and_fan_out
from mindspore import Tensor
import math
from cybertron.configs.bert import PretrainedConfig

activation_map = {'relu': nn.ReLU(), 'gelu': nn.GELU(False), 'gelu_approximate': nn.GELU(), 'swish': nn.SiLU()}


class Dense(nn.Dense):

    def __init__(self, in_channels, out_channels, weight_init=None, bias_init=None, has_bias=True, activation=None):
        if weight_init is None:
            weight_init = initializer(HeUniform(math.sqrt(5)), (out_channels, in_channels))
        if bias_init is None:
            fan_in, _ = _calculate_fan_in_and_fan_out((out_channels, in_channels))
            bound = 1 / math.sqrt(fan_in)
            bias_init = initializer(Uniform(bound), (out_channels))
        super().__init__(in_channels,
                         out_channels,
                         weight_init=weight_init,
                         bias_init=bias_init,
                         has_bias=has_bias,
                         activation=activation)


class Embedding(nn.Embedding):

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 use_one_hot=False,
                 embedding_table='normal',
                 dtype=mindspore.float32,
                 padding_idx=None):
        if embedding_table == 'normal':
            embedding_table = Normal(1.0)
        super().__init__(vocab_size, embedding_size, use_one_hot, embedding_table, dtype, padding_idx)

    @classmethod
    def from_pretrained_embedding(cls, embeddings: Tensor, freeze=True, padding_idx=None):
        rows, cols = embeddings.shape
        embedding = cls(rows, cols, embedding_table=embeddings, padding_idx=padding_idx)
        embedding.embedding_table.requires_grad = not freeze
        return embedding


class BertConfig(PretrainedConfig):
    """
    Configuration for BERT
    """

    def __init__(self,
                 vocab_size=30522,
                 hidden_size=768,
                 seq_len=40,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps


class BertEmbeddings(nn.Cell):
    """
    Embeddings for BERT, include word, position and token_type
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = Embedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm((config.hidden_size, ), epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(1 - config.hidden_dropout_prob)
        self.seq_len = config.seq_len

    def construct(self, input_ids, token_type_ids=None, position_ids=None):
        # seq_len = input_ids.shape[1]
        seq_len = self.seq_len
        if position_ids is None:
            position_ids = ops.arange(start=0,end=seq_len, step=1, dtype=mstype.int64)
            # position_ids = ops.zeros((seq_len),mstype.int64)
            # for i in range(seq_len):
            #     position_ids[i]=i
            position_ids = position_ids.expand_dims(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Cell):
    """Self attention layer for BERT
    """

    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention "
                             "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Dense(config.hidden_size, self.all_head_size)
        self.key = Dense(config.hidden_size, self.all_head_size)
        self.value = Dense(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(1 - config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.transpose(0, 2, 1, 3)

    def construct(self, hidden_states, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))
        attention_scores = attention_scores / ops.sqrt(ops.scalar_to_tensor(self.attention_head_size, mstype.float32))
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(axis=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = ops.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer, )
        return outputs


class BertSelfOutput(nn.Cell):

    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = Dense(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm((config.hidden_size, ), epsilon=1e-12)
        self.dropout = nn.Dropout(1 - config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Cell):

    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self_attn = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def construct(self, input_tensor, attention_mask=None, head_mask=None):
        self_outputs = self.self_attn(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output, ) + self_outputs[1:]
        return outputs


class BertIntermediate(nn.Cell):

    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = Dense(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = activation_map.get(config.hidden_act, nn.GELU(False))

    def construct(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Cell):

    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = Dense(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm((config.hidden_size, ), epsilon=1e-12)
        self.dropout = nn.Dropout(1 - config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Cell):

    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def construct(self, hidden_states, attention_mask=None, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output, ) + attention_outputs[1:]
        return outputs


class BertEncoder(nn.Cell):

    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.CellList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def construct(self, hidden_states, attention_mask=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states += (hidden_states, )

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions += (layer_outputs[1], )

        if self.output_hidden_states:
            all_hidden_states += (hidden_states, )

        outputs = (hidden_states, )
        if self.output_hidden_states:
            outputs += (all_hidden_states, )
        if self.output_attentions:
            outputs += (all_attentions, )
        return outputs


class BertPooler(nn.Cell):

    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = Dense(config.hidden_size, config.hidden_size, activation='tanh')

    def construct(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        return pooled_output


class BertModel(nn.Cell):
    """"""

    def __init__(self, config):
        super().__init__()
        self.config=config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.num_hidden_layers = self.config.num_hidden_layers

    def construct(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        if attention_mask is None:
            attention_mask = ops.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(input_ids)

        extended_attention_mask = attention_mask.expand_dims(1).expand_dims(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.ndim == 1:
                head_mask = head_mask.expand_dims(0).expand_dims(0).expand_dims(-1).expand_dims(-1)
                head_mask = ops.broadcast_to(head_mask, (self.num_hidden_layers, -1, -1, -1, -1))
            elif head_mask.ndim == 2:
                head_mask = head_mask.expand_dims(1).expand_dims(-1).expand_dims(-1)
        else:
            head_mask = [None] * self.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        # pooled_output = self.pooler(sequence_output)

        # outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        # return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
        return sequence_output
