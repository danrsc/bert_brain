from collections import OrderedDict
import itertools

import numpy as np
import torch
from torch import nn
from pytorch_pretrained_bert.modeling import gelu, BertLayerNorm

from ..common import NamedSpanEncoder
from .graph_part import GraphPart
from .grouping_modules import at_most_one_data_id, GroupConcatFixedGroupSize

__all__ = ['KeyedLinear', 'KeyedCombinedLinear', 'KeyedSingleTargetSpanAttention', 'group_concat_linear']


def group_concat_linear(
        name,
        num_per_group,
        groupby_prefixes,
        groupby_suffix,
        sequence_source_name,
        pooled_source_name=None,
        hidden_sizes=None,
        hidden_activation=gelu,
        force_cpu=False,
        output_key_to_shape=None,
        targets=None):
    result = OrderedDict()
    result['{}_group_concat'.format(name)] = GroupConcatFixedGroupSize(
        num_per_group,
        groupby_prefixes,
        groupby_suffix,
        'group_concat_fixed_group_size_output',
        sequence_source_name,
        pooled_source_name)
    result['{}_linear'.format(name)] = KeyedLinear(
        'group_concat_fixed_group_size_output',
        is_sequence=False,
        hidden_sizes=hidden_sizes,
        hidden_activation=hidden_activation,
        force_cpu=force_cpu,
        output_key_to_shape=output_key_to_shape,
        targets=targets)
    return result


class KeyedBase(GraphPart):

    def __init__(self, output_key_to_shape, targets):
        super().__init__()
        if output_key_to_shape is None:
            self.output_key_to_shape = OrderedDict()
        else:
            self.output_key_to_shape = OrderedDict(output_key_to_shape)
        self.splits = [int(np.prod(self.output_key_to_shape[k])) for k in self.output_key_to_shape]
        if targets is None:
            self.targets = set()
        elif np.ndim(targets) == 0:
            self.targets = {targets}
        else:
            self.targets = set(targets)

    def resolve_placeholders(self, placeholder_name_to_fields, field_shapes):
        if self.targets is not None:
            remaining_targets = set()
            for target in self.targets:
                if target in placeholder_name_to_fields:
                    for field in placeholder_name_to_fields[target]:
                        if field not in self.output_key_to_shape or self.output_key_to_shape[field] is None:
                            self.output_key_to_shape[field] = field_shapes[field]
                else:
                    remaining_targets.add(target)
            self.targets = remaining_targets
        self.splits = [int(np.prod(self.output_key_to_shape[k])) for k in self.output_key_to_shape]

    def forward(self, batch):
        raise NotImplementedError('{} does not implement forward'.format(type(self)))

    def instantiate(self, name_to_num_channels):
        if self.targets is not None:
            for target in self.targets:
                if target not in self.output_key_to_shape or self.output_key_to_shape[target] is None:
                    self.output_key_to_shape[target] = name_to_num_channels[target]
        for key in self.output_key_to_shape:
            if self.output_key_to_shape[key] is None:
                self.output_key_to_shape[key] = name_to_num_channels[key]
        if len(self.output_key_to_shape) == 0:
            raise ValueError('No outputs set')
        self.splits = [int(np.prod(self.output_key_to_shape[k])) for k in self.output_key_to_shape]
        return self._instantiate(name_to_num_channels)

    def _instantiate(self, name_to_num_channels):
        raise NotImplementedError('{} does not implement instantiate'.format(type(self)))

    def update_state_dict(self, prefix, state_dict, old_prediction_key_to_shape):
        old_splits = np.cumsum([int(np.prod(old_prediction_key_to_shape[k])) for k in old_prediction_key_to_shape])
        old_splits = dict((k, (0 if i == 0 else old_splits[i - 1], old_splits[i]))
                          for i, k in enumerate(old_prediction_key_to_shape))
        ranges = [old_splits[k] if k in old_splits else None for k in self.output_key_to_shape]
        for idx, k in enumerate(self.output_key_to_shape):
            if ranges[idx] is not None and \
                    int(np.prod(self.output_key_to_shape[k])) != ranges[idx][1] - ranges[idx][0]:
                raise ValueError('Inconsistent number of targets for prediction key: {}'.format(k))
        current_splits = [0] + np.cumsum(self.splits).tolist()
        total = current_splits[-1]
        current_splits = current_splits[:-1]

        def update(module, prefix_=''):
            for name, tensor in itertools.chain(
                    module.named_buffers(prefix_[:-1], False), module.named_parameters(prefix_[:-1], False)):
                if name in state_dict:
                    state = state_dict[name]
                    updated_state = tensor.clone()
                    for idx_split in range(len(current_splits)):
                        if ranges[idx_split] is not None:
                            end = current_splits[idx_split + 1] if idx_split + 1 < len(current_splits) else total
                            if len(state.size()) < 3:
                                updated_state[current_splits[idx_split]:end] = \
                                    state[ranges[idx_split][0]:ranges[idx_split][1]]
                            else:
                                raise ValueError('Unexpected state size: {}'.format(len(state.size())))
                    state_dict[name] = updated_state

            for name, child in module.named_children():
                if child is not None:
                    update(child, prefix_ + name + '.')

        update(self, prefix)


class _HiddenLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, activation_function=gelu, should_norm=True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.activation_function = activation_function
        self.layer_norm = BertLayerNorm(out_channels, eps=1e-12) if should_norm else None

    def forward(self, x):
        x = self.linear(x)
        if self.activation_function is not None:
            x = self.activation_function(x)
        if self.layer_norm is not None:
            return self.layer_norm(x)
        return x


class KeyedLinear(KeyedBase):

    def __init__(
            self,
            source_name,
            is_sequence,
            hidden_sizes=None,
            hidden_activation=gelu,
            force_cpu=False,
            output_key_to_shape=None,
            targets=None,
            apply_at_most_one_data_id=False):
        super().__init__(output_key_to_shape, targets)
        self.source_name = source_name
        self.is_sequence = is_sequence
        self.force_cpu = force_cpu
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.hidden = None
        self.linear = None
        self.apply_at_most_one_data_id = apply_at_most_one_data_id

    def _instantiate(self, name_to_num_channels):
        in_channels = name_to_num_channels[self.source_name]
        if self.hidden_sizes is not None:
            hidden_sizes = [self.hidden_sizes] if np.ndim(self.hidden_sizes) == 0 else self.hidden_sizes
            hidden_modules = list()
            for index_hidden in range(len(hidden_sizes)):
                current_in = in_channels if index_hidden == 0 else hidden_sizes[index_hidden - 1]
                hidden_modules.append(_HiddenLayer(current_in, hidden_sizes[index_hidden], self.hidden_activation))
            self.hidden = torch.nn.Sequential(*hidden_modules)
            in_channels = hidden_sizes[-1]
        self.linear = nn.Linear(in_channels, sum(self.splits))
        result = OrderedDict(self.output_key_to_shape)
        for key in name_to_num_channels:
            if isinstance(key, tuple) and key[0] == self.source_name:
                for result_key in self.output_key_to_shape:
                    result[(result_key,) + key[1:]] = name_to_num_channels[key]
        return result

    def forward(self, batch):
        x = batch[self.source_name]
        if self.hidden is not None:
            x = self.hidden(x)
        if self.force_cpu:
            x = x.cpu()
        predictions = self.linear(x)
        predictions = torch.split(predictions, self.splits, dim=-1)
        result = OrderedDict()
        assert(len(self.output_key_to_shape) == len(predictions))
        for k, p in zip(self.output_key_to_shape, predictions):
            for key in batch:
                if isinstance(key, tuple) and key[0] == self.source_name:
                    result[(k,) + key[1:]] = batch[key]
            if self.is_sequence:
                p = p.view(p.size()[:2] + self.output_key_to_shape[k])
            else:
                p = p.view(p.size()[:1] + self.output_key_to_shape[k])
            result[k] = p
            if (self.apply_at_most_one_data_id == 'if_no_target' and k not in batch) or self.apply_at_most_one_data_id:
                data_ids = at_most_one_data_id(batch[(k, 'data_ids')])
                indicator_valid = data_ids >= 0
                result[k] = result[k][indicator_valid]
                result[(k, 'data_ids')] = data_ids[indicator_valid]
                result[(k, 'example_ids')] = torch.arange(len(data_ids), device=data_ids.device)[indicator_valid]

        return result


class KeyedCombinedLinear(KeyedBase):

    def __init__(
            self,
            sequence_source_name,
            pooled_source_name,
            output_key_to_shape=None,
            targets=None):
        super().__init__(output_key_to_shape, targets)
        self.sequence_source_name = sequence_source_name
        self.pooled_source_name = pooled_source_name
        self.sequence_linear = None
        self.pooled_linear = None

    def _instantiate(self, name_to_num_channels):
        self.sequence_linear = nn.Linear(name_to_num_channels[self.sequence_source_name], sum(self.splits))
        self.pooled_linear = nn.Linear(name_to_num_channels[self.pooled_source_name], sum(self.splits))
        result = OrderedDict(self.output_key_to_shape)
        for key in name_to_num_channels:
            if isinstance(key, tuple) and (key[0] == self.sequence_source_name or key[0] == self.pooled_source_name):
                for result_key in self.output_key_to_shape:
                    result[(result_key,) + key[1:]] = name_to_num_channels[key]
        return result

    def forward(self, batch):
        predictions = self.sequence_linear(batch[self.sequence_source_name]) \
                      + torch.unsqueeze(self.pooled_linear(batch[self.pooled_source_name]), 1)
        predictions = torch.split(predictions, self.splits, dim=-1)
        result = OrderedDict()
        assert(len(self.output_key_to_shape) == len(predictions))
        for k, p in zip(self.output_key_to_shape, predictions):
            p = p.view(p.size()[:2] + self.output_key_to_shape[k])
            result[k] = p

        return result


class KeyedSingleTargetSpanAttention(KeyedBase):

    def __init__(
            self,
            num_spans,
            sequence_source_name,
            span_source_name,
            pooled_source_name=None,
            conv_hidden_channels=None,
            conv_hidden_kernel=1,
            output_key_to_shape=None,
            targets=None):
        super().__init__(output_key_to_shape, targets)
        self.num_spans = num_spans
        self.sequence_source_name = sequence_source_name
        self.span_source_name = span_source_name
        self.pooled_source_name = pooled_source_name
        self.conv_hidden_channels = conv_hidden_channels
        self.conv_hidden_kernel = conv_hidden_kernel
        self.conv_hidden = None
        self.linear = None
        self.attention_logits = None
        self.named_span_encoder = NamedSpanEncoder(range(num_spans))

    def _instantiate(self, name_to_num_channels):

        in_sequence_channels = name_to_num_channels[self.sequence_source_name]

        if self.conv_hidden_channels is not None and self.conv_hidden_channels > 0:
            self.conv_hidden = torch.nn.ModuleList()
            for _ in range(self.num_spans):
                if self.conv_hidden_kernel == 1:  # special case, use linear to avoid transpose
                    self.conv_hidden.append(torch.nn.Linear(in_sequence_channels, self.conv_hidden_channels))
                else:
                    self.conv_hidden.append(torch.nn.Conv1d(
                        in_sequence_channels,
                        self.conv_hidden_channels,
                        self.conv_hidden_kernel,
                        padding=(self.conv_hidden_kernel - 1) / 2))
        else:
            self.conv_hidden = None

        attention_input_channels = self.conv_hidden_channels if self.conv_hidden is not None else in_sequence_channels
        self.attention_logits = torch.nn.ModuleList()
        for _ in range(self.num_spans):
            self.attention_logits.append(torch.nn.Linear(attention_input_channels, 1))

        pooled_channels = name_to_num_channels[self.pooled_source_name] if self.pooled_source_name is not None else 0

        self.linear = torch.nn.Linear(
            pooled_channels + attention_input_channels * self.num_spans, sum(self.splits))

        result = OrderedDict(self.output_key_to_shape)
        for key in name_to_num_channels:
            if isinstance(key, tuple) and key[0] == self.source_name:
                for result_key in self.output_key_to_shape:
                    result[(result_key,) + key[1:]] = name_to_num_channels[key]
        return result

    def forward(self, batch):
        span_ids = batch[self.span_source_name]
        span_indicators = self.named_span_encoder.torch_span_indicators(span_ids)
        span_embeddings = list()
        if self.pooled_source_name is not None:
            span_embeddings.append(batch[self.pooled_source_name])
        for index_span, span_name in enumerate(span_indicators):
            span_indicator = torch.unsqueeze(span_indicators[span_name], dim=2)
            if self.conv_hidden is not None:
                conv_hidden = self.conv_hidden[index_span]
                if isinstance(conv_hidden, torch.nn.Linear):
                    attention_input = conv_hidden(batch[self.sequence_source_name])
                else:
                    attention_input = conv_hidden(
                        batch[self.sequence_source_name].transpose(1, 2))  # conv takes (batch, channels, seq)
                    attention_input = attention_input.transpose(2, 1).contiguous()  # back to (batch, seq, channels)
            else:
                attention_input = batch[self.sequence_source_name]
            attention_logits = self.attention_logits(attention_input)
            # this is how the huggingface code does a masked attention
            # noinspection PyTypeChecker
            span_mask = (1.0 - span_indicator) * -10000.0
            attention_probabilities = torch.nn.functional.softmax(attention_logits + span_mask, dim=-1)
            # -> (batch, channels)
            span_embeddings.append(torch.sum(attention_probabilities * attention_input, dim=1))
        prediction_input = torch.cat(span_embeddings, dim=2)
        predictions = self.linear(prediction_input)
        predictions = torch.split(predictions, self.splits, dim=-1)
        assert (len(self.output_key_to_shape) == len(predictions))
        result = OrderedDict()
        for k, p in zip(self.output_key_to_shape, predictions):
            p = p.view(p.size()[:1] + self.output_key_to_shape[k])
            result[k] = p
        return result
