from collections import OrderedDict
from typing import Union, Tuple, Sequence, Optional, Callable

import numpy as np
from torch import Tensor, nn

from .gelu_new_module import gelu_new as gelu

from .graph_part import GraphPart


__all__ = ['MultiLayerBottleneck']


def multi_layer_bottleneck_layer_norm(x: Tensor, eps, weight, bias, axis=-2):
    # note we norm along the second to last dimension
    mean = x.mean(dim=axis, keepdim=True)
    std = x.std(dim=axis, keepdim=True)
    x = (x - mean) / (std + eps)
    if weight is not None:
        x = weight * x
    if bias is not None:
        x = x + bias
    return x


class _MultiBottleneckLayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5, elementwise_affine=True, channels_last=False):
        super().__init__()
        self.channels_last = channels_last
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            if self.channels_last:
                self.weight = nn.Parameter(Tensor(num_channels))
                self.bias = nn.Parameter(Tensor(num_channels))
            else:
                self.weight = nn.Parameter(Tensor(num_channels, 1))
                self.bias = nn.Parameter(Tensor(num_channels, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor):
        return multi_layer_bottleneck_layer_norm(
            x, self.eps, self.weight, self.bias, axis=-1 if self.channels_last else -2)


class _BottleneckCombineWithLayerNormModule(nn.Module):

    def __init__(
            self,
            num_channels,
            num_input_bottlenecks,
            num_output_bottlenecks,
            activation_function=gelu,
            should_norm=True,
            should_transpose_output=False):
        super().__init__()
        self._should_squeeze = num_output_bottlenecks == 0
        if self._should_squeeze:
            num_output_bottlenecks = 1
        self.linear = nn.Linear(num_input_bottlenecks, num_output_bottlenecks)
        self.activation_function = activation_function
        self.should_transpose_output = should_transpose_output
        self.layer_norm = None
        if should_norm:
            self.layer_norm = _MultiBottleneckLayerNorm(num_channels)

    def forward(self, x):
        x = self.linear(x)
        if self.activation_function is not None:
            x = self.activation_function(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        if self._should_squeeze:
            x = x.squeeze(-1)
        elif self.should_transpose_output:
            x = x.transpose(-2, -1)
        return x


class MultiLayerBottleneck(GraphPart):

    def __init__(
            self,
            source_name: Union[str, Tuple[str, ...]],
            output_name,
            num_bottleneck_channels: int,
            num_output_bottlenecks: int,
            num_hidden_bottlenecks: Optional[Union[int, Sequence[int]]] = None,
            hidden_activation: Optional[Callable[[Tensor], Tensor]] = gelu,
            should_norm_hidden: bool = True,
            should_norm: bool = False,
            should_transpose_output: bool = True):
        super().__init__()
        self.source_name = source_name
        self.output_name = output_name
        self.num_bottleneck_channels = num_bottleneck_channels
        self.num_output_bottlenecks = num_output_bottlenecks
        self.num_hidden_bottlenecks = num_hidden_bottlenecks
        self.hidden_activation = hidden_activation
        self.should_norm_hidden = should_norm_hidden
        self.bottleneck_combine = None
        self.linear = None
        self.single_layer_norm = None
        self.should_norm = should_norm
        self.should_transpose_output = should_transpose_output

    def resolve_placeholders(self, placeholder_name_to_fields, field_shapes, num_response_data_fields):
        pass

    @property
    def _num_first_layer_bottlenecks(self):
        if self.num_hidden_bottlenecks is None:
            return self.num_output_bottlenecks
        elif np.ndim(self.num_hidden_bottlenecks) == 0:
            return self.num_hidden_bottlenecks
        else:
            return self.num_hidden_bottlenecks[0]

    def instantiate(self, name_to_num_channels):
        in_channels = name_to_num_channels[self.source_name]
        num_hidden_bottlenecks = [] \
            if self.num_hidden_bottlenecks is None \
            else (
                [self.num_hidden_bottlenecks] if np.ndim(self.num_hidden_bottlenecks) == 0
                else self.num_hidden_bottlenecks)
        num_output_bottlenecks = num_hidden_bottlenecks + [self.num_output_bottlenecks]
        num_first = num_output_bottlenecks[0]
        if len(num_output_bottlenecks) == 1 and num_first == 0:
            num_first = 1  # 0 as the number of output bottlenecks is same as 1 but with squeeze
        self.linear = nn.Linear(in_channels, num_first * self.num_bottleneck_channels)
        bottleneck_combine = list()
        for idx_current in range(1, len(num_output_bottlenecks)):
            bottleneck_combine.append(_BottleneckCombineWithLayerNormModule(
                self.num_bottleneck_channels,
                num_output_bottlenecks[idx_current - 1],
                num_output_bottlenecks[idx_current],
                self.hidden_activation if idx_current < len(num_output_bottlenecks) - 1 else None,
                self.should_norm_hidden if idx_current < len(num_output_bottlenecks) - 1 else self.should_norm,
                should_transpose_output=False
                if idx_current < len(num_output_bottlenecks) - 1 else self.should_transpose_output))
        if len(bottleneck_combine) > 0:
            self.bottleneck_combine = nn.Sequential(*bottleneck_combine)
        elif self.should_norm:
            assert(len(num_output_bottlenecks) == 1)
            if num_output_bottlenecks[0] == 0:
                self.single_layer_norm = nn.LayerNorm(self.num_bottleneck_channels)
            elif self.should_transpose_output:
                self.single_layer_norm = _MultiBottleneckLayerNorm(self.num_bottleneck_channels, channels_last=True)
            else:
                self.single_layer_norm = _MultiBottleneckLayerNorm(self.num_bottleneck_channels)
        result = OrderedDict()
        num_out = num_output_bottlenecks[-1]
        if num_out == 0:
            # 0 as the number of output bottlenecks is same as 1 but with squeeze
            num_out = self.num_bottleneck_channels
        elif self.should_transpose_output:
            num_out = num_out, self.num_bottleneck_channels
        else:
            num_out = self.num_bottleneck_channels, num_out
        result[self.output_name] = num_out
        for key in name_to_num_channels:
            if isinstance(key, tuple) and key[0] == self.source_name:
                result[(self.output_name,) + key[1:]] = name_to_num_channels[key]
        return result

    def forward(self, batch):
        x = batch[self.source_name]
        if self.bottleneck_combine is not None:
            x = self.linear(x).view(x.size()[:-1] + (self.num_bottleneck_channels, self._num_first_layer_bottlenecks))
            x = self.bottleneck_combine(x)
        else:
            x = self.linear(x)
            if self._num_first_layer_bottlenecks != 0:
                if self.should_transpose_output:
                    x = x.view(x.size()[:-1] + (self._num_first_layer_bottlenecks, self.num_bottleneck_channels))
                else:
                    x = x.view(x.size()[:-1] + (self.num_bottleneck_channels, self._num_first_layer_bottlenecks))
            if self.single_layer_norm is not None:
                x = self.single_layer_norm(x)

        result = OrderedDict()
        for key in batch:
            if isinstance(key, tuple) and key[0] == self.source_name:
                result[(self.output_name,) + key[1:]] = batch[key]
        result[self.output_name] = x

        return result
