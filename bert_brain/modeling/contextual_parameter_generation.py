from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torch.nn.functional

from pytorch_pretrained_bert.modeling import BertLayerNorm

from .graph_part import GraphPart


__all__ = ['LinearContextualParameterGeneration']


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


class ContextualizedBertLayerNorm(nn.Module):
    def __init__(self, bert_layer_norm):
        super().__init__()
        self.weight = bert_layer_norm.weight.detach()
        self.bias = bert_layer_norm.bias.detach()
        self.variance_epsilon = bert_layer_norm.variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


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
                elif type(module_) is BertLayerNorm:
                    replacement_module = ContextualizedBertLayerNorm(module_)
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

        # detach the variables in the contextualized modules
        for name in self._generated_parameters:
            module, attr_name, shape = self._generated_parameters[name]
            setattr(module, attr_name, getattr(module, attr_name).detach())

        return outputs
