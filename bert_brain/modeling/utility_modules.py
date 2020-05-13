from collections import OrderedDict
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
import torch.nn
from torch import nn
from .gelu_new_module import gelu_new as gelu

from .graph_part import GraphPart, GraphPartFactory


__all__ = [
    'Conv1DCausal',
    'PooledFromSequenceFactory',
    'PooledFromSequence',
    'PooledFromKTokensFactory',
    'PooledFromKTokens',
    'LinearWithLayerNorm',
    'QuasiAttention',
    'HiddenReconstructionPenalty']


@dataclass(frozen=True)
class PooledFromSequenceFactory(GraphPartFactory):
    source_name: str
    output_name: str
    transform_fn: Any = None

    def make_graph_part(self):
        return PooledFromSequence(self.source_name, self.output_name, self.transform_fn)


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


@dataclass(frozen=True)
class PooledFromKTokensFactory(GraphPartFactory):
    num_tokens: int
    source_name: str
    output_name: str
    transform_fn: Any = None
    use_first_k: bool = False

    def make_graph_part(self):
        return PooledFromKTokens(
            self.num_tokens, self.source_name, self.output_name, self.transform_fn, self.use_first_k)


class PooledFromKTokens(GraphPart):

    def __init__(self, num_tokens, source_name, output_name, transform_fn=None, use_first_k=False):
        super().__init__()
        self.num_tokens = num_tokens
        self.use_first_k = use_first_k
        self.source_name = source_name
        self.output_name = output_name
        self.linear = None
        self.transform_fn = transform_fn

    def resolve_placeholders(self, placeholder_name_to_fields, field_shapes, num_response_data_fields):
        pass

    def instantiate(self, name_to_num_channels):
        result = OrderedDict()
        result[self.output_name] = name_to_num_channels[self.source_name]
        self.linear = nn.Linear(self.num_tokens, 1)
        for key in name_to_num_channels:
            if isinstance(key, tuple) and key[0] == self.source_name:
                result[(self.output_name,) + key[1:]] = name_to_num_channels[key]
        return result

    def forward(self, batch):
        result = OrderedDict()
        if self.use_first_k:
            x = batch[self.source_name][:, :self.num_tokens]
        else:
            x = batch[self.source_name][:, -self.num_tokens:]
        pad_size = self.num_tokens - x.size(1)
        if pad_size > 0:
            if self.use_first_k:
                x = torch.nn.functional.pad(x, [0, pad_size, 0, 0])
            else:
                x = torch.nn.functional.pad(x, [pad_size, 0, 0, 0])
        x = torch.transpose(x, -2, -1)
        x = torch.squeeze(self.linear(x), -1)
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


class LinearWithLayerNorm(torch.nn.Module):

    def __init__(self, in_channels, out_channels, bias=True, activation_function=gelu, should_norm=True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias)
        self.activation_function = activation_function
        self.layer_norm = torch.nn.LayerNorm(out_channels) if should_norm else None

    def forward(self, x):
        x = self.linear(x)
        if self.activation_function is not None:
            x = self.activation_function(x)
        if self.layer_norm is not None:
            return self.layer_norm(x)
        return x


class QuasiAttention(torch.nn.Module):

    def __init__(self, in_channels, out_channels, bias=True, activation_function=gelu, should_norm=False):
        super().__init__()
        self.attention = nn.Parameter(torch.randn(in_channels, out_channels, requires_grad=True), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, requires_grad=True), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.activation_function = activation_function
        self.layer_norm = torch.nn.LayerNorm(out_channels) if should_norm else None

    def forward(self, x):
        weights = nn.functional.softmax(self.attention, 0)
        x = torch.matmul(x, weights)
        if self.bias is not None:
            x = x + self.bias
        if self.activation_function is not None:
            x = self.activation_function(x)
        if self.layer_norm is not None:
            return self.layer_norm(x)
        return x


class HiddenReconstructionPenalty(torch.nn.Module):

    def __init__(
            self,
            penultimate_reconstruction_penalty_coefficient: float,
            penultimate_reconstruction_l1_weight_coefficient: float,
            penultimate_reconstruction_output_name: str = 'rcn',
            activation_fn=None,
            should_norm=False):
        super().__init__()
        self.penultimate_reconstruction_output_name = penultimate_reconstruction_output_name
        self.penultimate_reconstruction_penalty_coefficient = penultimate_reconstruction_penalty_coefficient
        self.penultimate_reconstruction_l1_weight_coefficient = penultimate_reconstruction_l1_weight_coefficient
        if self.penultimate_reconstruction_penalty_coefficient < 0:
            raise ValueError('penultimate_reconstruction_penalty_coefficient must be >= 0')
        if self.penultimate_reconstruction_l1_weight_coefficient < 0:
            raise ValueError('penultimate_reconstruction_l1_weight_coefficient must be >= 0')
        if self.penultimate_reconstruction_penalty_coefficient == 0 \
                and self.penultimate_reconstruction_l1_weight_coefficient > 0:
            raise ValueError('penultimate_reconstruction_l1_weight_coefficient can only be > 0 '
                             'if penultimate_reconstruction_penalty_coefficient > 0')
        self.activation_fn = activation_fn
        self.should_norm = should_norm
        self.reconstruction_linear = None

    def instantiate(self, in_channels, out_channels):
        if self.penultimate_reconstruction_penalty_coefficient > 0:
            self.reconstruction_linear = LinearWithLayerNorm(
                in_channels, out_channels, self.activation_fn, self.should_norm)

    def forward(self, source, target):
        result = OrderedDict()
        if self.reconstruction_linear is not None:
            reconstruction = self.reconstruction_linear(source)
            result[self.penultimate_reconstruction_output_name] = reconstruction - target
        return result

    def compute_penalties(self, batch, predictions, loss_dict):
        result = OrderedDict()
        if self.penultimate_reconstruction_l1_weight_coefficient > 0:
            # weighting by the loss is problematic for 2 reasons:
            # 1) the losses can have different scales (1000-way classification vs. binary vs. mse)
            # 2) it is unclear how to handle the absence of a loss in a batch...set to 1?

            # split_weights = torch.split(self.reconstruction_linear.linear.weight, self.splits, -1)
            # assert(len(split_weights) == len(self.output_key_to_shape))
            # for k, w in zip(self.output_key_to_shape, split_weights):
            #     if k in loss_dict:
            #         loss_weight, loss = loss_dict[k]
            #         no_valid_inputs = isinstance(loss, str) and loss == 'no_valid_inputs'
            #         if not no_valid_inputs:
            #             scale = loss * self.penultimate_reconstruction_l1_weight_coefficient
            #             result['l1_{}_{}'.format(self.penultimate_reconstruction_output_name, k)] = \
            #                 loss_weight, scale * torch.sum(torch.abs(w))

            # for now we just use a total l1
            result['l1_{}'.format(self.penultimate_reconstruction_output_name)] = (
                1,
                self.penultimate_reconstruction_l1_weight_coefficient * torch.sum(
                    torch.abs(self.reconstruction_linear.linear.weight)))
        if self.penultimate_reconstruction_penalty_coefficient > 0:
            result[self.penultimate_reconstruction_output_name] = (
                1,
                self.penultimate_reconstruction_penalty_coefficient
                * torch.sum(predictions[self.penultimate_reconstruction_output_name] ** 2)
                / predictions[self.penultimate_reconstruction_output_name].size()[0])
        return result if len(result) > 0 else None
