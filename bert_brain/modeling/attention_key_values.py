from collections import OrderedDict
from typing import Union, Tuple

from torch import nn

from .graph_part import GraphPart


__all__ = ['AttentionKeyValues']


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
