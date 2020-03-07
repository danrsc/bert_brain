from collections import OrderedDict
from typing import Union, Tuple

import numpy as np
import torch
from torch import nn

from .graph_part import GraphPart


__all__ = ['AttentionKeyValues', 'AttentionPool']


class AttentionKeyValues(GraphPart):

    def __init__(
            self,
            source_name: Union[str, Tuple[str, ...]],
            output_key_name: str,
            output_value_name: str,
            num_heads: int,
            num_key_channels: int,
            num_value_channels: int,
            value_activation_fn=None):
        super().__init__()
        self.source_name = source_name
        self.output_key_name = output_key_name
        self.output_value_name = output_value_name
        self.num_heads = num_heads
        self.num_key_channels = num_key_channels
        self.num_value_channels = num_value_channels
        self.key_linear = None
        self.value_linear = None
        self.value_activation_fn = value_activation_fn

    def resolve_placeholders(self, placeholder_name_to_fields, field_shapes, num_response_data_fields):
        pass

    def instantiate(self, name_to_num_channels):
        in_channels = name_to_num_channels[self.source_name]
        self.key_linear = nn.Linear(in_channels, self.num_heads * self.num_key_channels)
        self.value_linear = nn.Linear(in_channels, self.num_heads * self.num_value_channels)
        result = OrderedDict()
        result[self.output_key_name] = (self.num_heads, self.num_key_channels)
        result[self.output_value_name] = (self.num_heads, self.num_value_channels)
        for key in name_to_num_channels:
            if isinstance(key, tuple) and key[0] == self.source_name:
                result[(self.output_value_name,) + key[1:]] = name_to_num_channels[key]
        return result

    def forward(self, batch):
        key = self.key_linear(batch[self.source_name])
        value = self.value_linear(batch[self.source_name])
        key = key.view(key.size()[:-1] + (self.num_heads, self.num_key_channels))
        value = value.view(value.size()[:-1] + (self.num_heads, self.num_value_channels))
        if self.value_activation_fn is not None:
            value = self.value_activation_fn(value)
        result = OrderedDict()
        for k in batch:
            if isinstance(k, tuple) and k[0] == self.source_name:
                result[(self.output_key_name,) + k[1:]] = batch[k]
        result[self.output_key_name] = key
        result[self.output_value_name] = value
        return result


class AttentionPool(GraphPart):

    def __init__(self, key_name, value_name, output_name, should_layer_norm=False, flatten=False):
        super().__init__()
        self.key_name = key_name
        self.value_name = value_name
        self.output_name = output_name
        self.should_layer_norm = should_layer_norm
        self.flatten = flatten
        self.query = None
        self.layer_norm = None

    def resolve_placeholders(self, placeholder_name_to_fields, field_shapes, num_response_data_fields):
        pass

    def instantiate(self, name_to_num_channels):
        num_heads, key_channels = name_to_num_channels[self.key_name]
        num_heads, value_channels = name_to_num_channels[self.value_name]
        self.query = nn.Parameter(torch.randn(num_heads, key_channels, requires_grad=True), requires_grad=True)
        result = OrderedDict()
        output_shape = num_heads * value_channels if self.flatten else (num_heads, value_channels)
        if self.should_layer_norm:
            self.layer_norm = nn.LayerNorm(output_shape)
        result[self.output_name] = output_shape
        for key in name_to_num_channels:
            if isinstance(key, tuple) and key[0] == self.value_name:
                result[(self.output_name,) + key[1:]] = name_to_num_channels[key]
        return result

    def forward(self, batch):
        # (heads, key_channels)
        query = self.query
        # (batch, sequence, ..., heads, key_channels)
        key = batch[self.key_name]
        # -> (batch, ..., heads, key_channels, sequence)
        key = key.permute([0] + list(range(2, len(key.size()) - 1)) + [-1, 1])
        # -> (1, ..., heads, key_channels)
        while len(query.size()) < len(key.size()) - 1:
            query = torch.unsqueeze(query, 0)
        # -> (1, ..., heads, 1, key_channels)
        query = torch.unsqueeze(query, -2)
        # -> (batch, ..., heads, 1, sequence)
        attention_scores = torch.matmul(query, key)
        attention_scores = attention_scores / np.sqrt(key.size()[-1])
        # apply softmax over the sequence
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # -> (batch, sequence, ..., heads, 1)
        attention_probs = attention_probs.permute([0, -1] + list(range(1, len(attention_probs.size()) - 1)))
        # (batch, sequence, ..., heads, value_channels)
        value = batch[self.value_name]
        # (batch, ..., heads, value_channels)
        output = torch.sum(attention_probs * value, dim=1)
        if self.flatten:
            output = output.view(output.size()[:-2] + (output.size()[-2] * output.size()[-1],))
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        result = OrderedDict()
        for key in batch:
            if isinstance(key, tuple) and key[0] == self.value_name:
                result[(self.output_name,) + key[1:]] = batch[key]
        result[self.output_name] = output
        return result
