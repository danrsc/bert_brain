from collections import OrderedDict
from typing import Optional, Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional

from .graph_part import GraphPart


__all__ = ['LinearContextualParameterGeneration', 'ContextualBottleneckSum', 'ContextAttention',
           'LinearDecreasingTemperatureSchedule']


class ContextualizedLinear(nn.Module):
    # copied from torch.nn.Linear and modified to copy parameters from the non-contextualized layer into variables
    # and then to use those variables
    def __init__(self, linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight.detach()
        self.bias = linear.bias.detach() if linear.bias is not None else None

    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias)


class ContextualizedLayerNorm(nn.Module):
    def __init__(self, layer_norm):
        super().__init__()
        self.weight = layer_norm.weight.detach()
        self.bias = layer_norm.bias.detach()
        self.eps = layer_norm.eps
        self.normalized_shape = layer_norm.normalized_shape

    def forward(self, x):
        return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class LinearContextualParameterGeneration(GraphPart):

    def __init__(
            self,
            context_id_source_name,
            num_contexts,
            embedding_size,
            inner_graph_parts: OrderedDict):
        super().__init__()
        self.context_id_source_name = context_id_source_name
        self.num_contexts = num_contexts
        self.embedding_size = embedding_size
        self.inner_graph_parts = OrderedDict(inner_graph_parts)
        self._generated_parameters = None
        self._splits = None
        self.generator = None
        self.embedding = None

    def resolve_placeholders(self, placeholder_name_to_fields, field_shapes, num_response_data_fields):
        if self.num_contexts == 'num_response_data_fields':
            self.num_contexts = num_response_data_fields
        for key in self.inner_graph_parts:
            self.inner_graph_parts[key].resolve_placeholders(
                placeholder_name_to_fields, field_shapes, num_response_data_fields)

    def instantiate(self, name_to_num_channels):
        # noinspection PyTypeChecker
        for key in self.inner_graph_parts:
            # noinspection PyUnresolvedReferences
            graph_part_num_channels = self.inner_graph_parts[key].instantiate(name_to_num_channels)
            for k in graph_part_num_channels:
                if k in name_to_num_channels:
                    raise ValueError('Duplicate output: {}'.format(k))
                name_to_num_channels[k] = graph_part_num_channels[k]

        # noinspection PyTypeChecker
        self.inner_graph_parts = torch.nn.ModuleDict(
            modules=[(k, self.inner_graph_parts[k]) for k in self.inner_graph_parts])

        def gather_parameters(result, module_, prefix_=''):
            module_result = OrderedDict()
            for name_, tensor in module_.named_parameters(prefix_[:-1], False):
                if name_ in result:
                    raise ValueError('Duplicate name in parameters: {}'.format(name_))
                if tensor is not None:
                    module_result[name_] = (module_, name_[len(prefix_):], tensor.size())

            # this is very ugly, but for reasonable reasons, PyTorch does not allow the modification of
            # parameters in a module into variables, so we create specialized modules. We could support
            # multiple types here easily, but for now we just support nn.Linear
            if len(module_result) > 0:
                if type(module_) is nn.Linear:
                    replacement_module = ContextualizedLinear(module_)
                elif type(module_) is nn.LayerNorm:
                    replacement_module = ContextualizedLayerNorm(module_)
                else:
                    raise ValueError('Unsupported module type: {}'.format(type(module_)))
                for name_ in module_result:
                    result[name_] = (replacement_module,) + module_result[name_][1:]
                assert (len(list(module_.named_children())) == 0)
                return replacement_module

            replacements = dict()
            for name_, child in module_.named_children():
                if child is not None:
                    child_replacement = gather_parameters(result, child, prefix_ + name_ + '.')
                    if child_replacement is not None:
                        replacements[name_] = child_replacement

            for name_ in replacements:
                setattr(module_, name_, replacements[name_])

        self._generated_parameters = OrderedDict()
        for key in self.inner_graph_parts:
            gather_parameters(self._generated_parameters, self.inner_graph_parts[key])

        self._splits = [int(np.prod(self._generated_parameters[k][2])) for k in self._generated_parameters]

        self.generator = torch.nn.Linear(self.embedding_size, sum(self._splits))
        self.embedding = torch.nn.Embedding(self.num_contexts, self.embedding_size, max_norm=1)

        return name_to_num_channels

    def forward(self, batch):
        context_id = batch[self.context_id_source_name]
        # noinspection PyTypeChecker
        if not torch.all(context_id == context_id[0]):
            raise ValueError('Expected a single context_id per batch')
        context_embedding = self.embedding(context_id[0])
        parameters = self.generator(context_embedding)
        parameters = torch.split(parameters, self._splits, dim=-1)
        assert(len(parameters) == len(self._generated_parameters))
        for name, parameter in zip(self._generated_parameters, parameters):
            module, attr_name, shape = self._generated_parameters[name]
            setattr(module, attr_name, parameter.view(shape))

        outputs = OrderedDict()
        for name in self.inner_graph_parts:
            graph_part = self.inner_graph_parts[name]
            graph_part_outputs = graph_part(batch)

            for k in graph_part_outputs:
                if k in outputs:  # don't allow two graph parts to output the same key
                    raise ValueError('multiple predictions made for key: {}'.format(k))
                else:
                    outputs[k] = graph_part_outputs[k]
                    # overwrite batch[k] if it exists
                    batch[k] = graph_part_outputs[k]

        return outputs


@dataclass(frozen=True)
class LinearDecreasingTemperatureSchedule:
    start_temp: float
    num_steps: int

    def __call__(self, global_step):
        return max((1 - self.start_temp) * global_step / self.num_steps, 1)


class ContextualBottleneckSum(GraphPart):

    def __init__(
            self,
            context_id_source_name,
            num_contexts,
            bottleneck_source_name,
            output_name,
            softmax_weights=False,
            softmax_temperature_schedule_fn: Optional[Callable[[int], float]]=None):

        super().__init__()
        self.context_id_source_name = context_id_source_name
        self.num_contexts = num_contexts
        self.bottleneck_source_name = bottleneck_source_name
        self.output_name = output_name
        self.embedding = None
        self.softmax = nn.Softmax(dim=-1) if softmax_weights else None
        self.softmax_temperature_schedule_fn = softmax_temperature_schedule_fn

    def resolve_placeholders(self, placeholder_name_to_fields, field_shapes, num_response_data_fields):
        if self.num_contexts == 'num_response_data_fields':
            self.num_contexts = num_response_data_fields

    def instantiate(self, name_to_num_channels):
        input_shape = name_to_num_channels[self.bottleneck_source_name]
        if np.ndim(input_shape) == 0:
            raise ValueError('Expected a 1d shape from {}'.format(self.bottleneck_source_name))
        if len(input_shape) != 2:
            raise ValueError('Expected a shape with len 2 from {}'.format(self.bottleneck_source_name))
        num_channels, num_bottlenecks = input_shape
        self.embedding = torch.nn.Embedding(
            self.num_contexts, num_bottlenecks, max_norm=None if self.softmax is not None else 1)
        result = OrderedDict()
        result[self.output_name] = num_channels
        for key in name_to_num_channels:
            if isinstance(key, tuple) and key[0] == self.source_name:
                result[(self.output_name,) + key[1:]] = name_to_num_channels[key]
        return result

    def forward(self, batch):
        context_id = batch[self.context_id_source_name]
        # noinspection PyTypeChecker
        if not torch.all(context_id == context_id[0]):
            raise ValueError('Expected a single context_id per batch')
        context_embedding = self.embedding(context_id[0])
        result = OrderedDict()
        for key in batch:
            if isinstance(key, tuple) and key[0] == self.bottleneck_source_name:
                result[(self.output_name,) + key[1:]] = batch[key]
        if self.softmax is not None:
            context_embedding = self.softmax(context_embedding / np.sqrt(context_embedding.size()[-1]))
            if self.softmax_temperature_schedule_fn is not None:
                temperature = self.softmax_temperature_schedule_fn(batch['global_step'])
                context_embedding = torch.pow(context_embedding, 1 / temperature)
                context_embedding = context_embedding / torch.sum(context_embedding)
        result[self.output_name] = nn.functional.linear(batch[self.bottleneck_source_name], context_embedding)
        return result


class ContextAttention(GraphPart):

    def __init__(self, context_id_source_name, num_contexts, key_name, value_name, output_name):
        super().__init__()
        self.context_id_source_name = context_id_source_name
        self.num_contexts = num_contexts
        self.key_name = key_name
        self.value_name = value_name
        self.output_name = output_name
        self.embedding = None
        self.softmax = nn.Softmax(dim=-1)

    def resolve_placeholders(self, placeholder_name_to_fields, field_shapes, num_response_data_fields):
        if self.num_contexts == 'num_response_data_fields':
            self.num_contexts = num_response_data_fields

    def instantiate(self, name_to_num_channels):
        num_heads, key_channels = name_to_num_channels[self.key_name]
        num_heads, value_channels = name_to_num_channels[self.value_name]
        self.embedding = torch.nn.Embedding(self.num_contexts, key_channels)
        result = OrderedDict()
        result[self.output_name] = value_channels
        for key in name_to_num_channels:
            if isinstance(key, tuple) and key[0] == self.value_name:
                result[(self.output_name,) + key[1:]] = name_to_num_channels[key]
        return result

    def forward(self, batch):
        # (batch, key_channels)
        query = self.embedding(batch[self.context_id_source_name])
        # (batch, ..., nodes, key_channels)
        key = batch[self.key_name]
        # -> (batch, ..., 1, key_channels)
        while len(query.size()) < len(key.size()):
            query = torch.unsqueeze(query, 1)
        # -> (batch, ..., nodes)
        attention_scores = torch.matmul(query, torch.transpose(key, -2, -1))
        attention_scores = attention_scores / np.sqrt(key.size()[-1])
        attention_probs = self.softmax(attention_scores)
        # (batch, ..., nodes, value_channels)
        value = batch[self.value_name]
        output = torch.matmul(attention_probs, value)
        # -> (batch, ..., value_channels)
        output = torch.squeeze(output, dim=-2)
        result = OrderedDict()
        for key in batch:
            if isinstance(key, tuple) and key[0] == self.value_name:
                result[(self.output_name,) + key[1:]] = batch[key]
        result[self.output_name] = output
        return result
