from collections import OrderedDict
import numpy as np
import torch
import torch.nn
from .graph_part import GraphPart


__all__ = ['Conv1DCausal', 'PooledFromSequence']


class PooledFromSequence(GraphPart):

    def __init__(self, source_name, output_name, transform_fn=None):
        super().__init__()
        self.source_name = source_name
        self.output_name = output_name
        self.transform_fn = transform_fn

    def resolve_placeholders(self, placeholder_name_to_fields, field_shapes, num_response_data_fields):
        pass

    def instantiate(self, name_to_num_channels):
        result = OrderedDict()
        result[self.output_name] = name_to_num_channels[self.source_name]
        for key in name_to_num_channels:
            if isinstance(key, tuple) and key[0] == self.source_name:
                result[(self.output_name,) + key[1:]] = name_to_num_channels[key]
        return result

    def forward(self, batch):
        result = OrderedDict()
        x = batch[self.source_name][:, 0]
        result[self.output_name] = x if self.transform_fn is None else self.transform_fn(x)
        for key in batch:
            if isinstance(key, tuple) and key[0] == self.source_name:
                result[(self.output_name,) + key[1:]] = batch[key]
        return result


class Conv1DCausal(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True,
                 transpose_axes=None, should_transpose_input=True, should_transpose_output=True):
        super().__init__()
        self.transpose_axes = transpose_axes
        self.should_transpose_input = should_transpose_input
        self.should_transpose_output = should_transpose_output
        padding = dilation * (kernel_size - 1)
        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        if self.transpose_axes is not None and self.should_transpose_input:
            x = x.permute(*self.transpose_axes)
        result = self.conv1d(x)
        # remove the element from the right padding
        result = result[:, :, :-self.conv1d.padding[0]]
        if self.transpose_axes is not None and self.should_transpose_output:
            result = result.permute(*np.argsort(self.transpose_axes))
        return result
