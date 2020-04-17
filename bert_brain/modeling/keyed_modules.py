from collections import OrderedDict
from typing import Optional, Union, Sequence, Callable, Mapping, Tuple, Iterable

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional

from transformers.modeling_bert import gelu_new as gelu

from .utility_modules import LinearWithLayerNorm, HiddenReconstructionPenalty, QuasiAttention
from ..common import NamedSpanEncoder
from .graph_part import GraphPart
from .grouping_modules import at_most_one_data_id, GroupConcatFixedGroupSize


__all__ = [
    'KeyedLinear',
    'KeyedQuasiAttention',
    'KeyedCombinedLinear',
    'KeyedSingleTargetSpanAttention',
    'KeyedSingleTargetSpanMaxPool',
    'group_concat_linear',
    'KeyedConcat']


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
        hidden_sizes=hidden_sizes,
        hidden_activation=hidden_activation,
        force_cpu=force_cpu,
        output_key_to_shape=output_key_to_shape,
        targets=targets)
    return result


class KeyedBase(GraphPart):

    @staticmethod
    def _output_shape_normalize(output_key_to_shape):
        keys = list(output_key_to_shape)
        for key in keys:
            if np.ndim(output_key_to_shape[key]) == 0:
                output_key_to_shape[key] = (output_key_to_shape[key],)
        return [int(np.prod(output_key_to_shape[k])) for k in output_key_to_shape]

    def __init__(
            self,
            output_key_to_shape: Optional[Mapping[str, Union[int, Tuple[int, ...]]]],
            targets: Optional[Union[str, Iterable[str]]]):
        super().__init__()
        if output_key_to_shape is None:
            self.output_key_to_shape = OrderedDict()
        else:
            self.output_key_to_shape = OrderedDict(output_key_to_shape)
        self.splits = KeyedBase._output_shape_normalize(self.output_key_to_shape)
        if targets is None:
            self.targets = set()
        elif np.ndim(targets) == 0:
            self.targets = {targets}
        else:
            self.targets = set(targets)

    def resolve_placeholders(self, placeholder_name_to_fields, field_shapes, num_response_data_fields):
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
        self.splits = KeyedBase._output_shape_normalize(self.output_key_to_shape)

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


class KeyedLinear(KeyedBase):

    def __init__(
            self,
            source_name: Union[str, Tuple[str, ...]],
            hidden_sizes: Optional[Union[int, Sequence[int]]] = None,
            hidden_activation: Optional[Callable[[Tensor], Tensor]] = gelu,
            should_norm_hidden: bool = True,
            force_cpu: bool = False,
            output_key_to_shape: Optional[Mapping[str, Union[int, Tuple[int, ...]]]] = None,
            targets: Optional[Union[str, Iterable[str]]] = None,
            apply_at_most_one_data_id: Union[str, bool, Mapping[str, Union[str, bool]]] = False,
            should_norm: bool = False,
            activation_fn: Optional[Callable[[Tensor], Tensor]] = None,
            penultimate_reconstruction_penalty_coefficient: float = 0,
            penultimate_reconstruction_l1_weight_coefficient: float = 0,
            penultimate_reconstruction_output_name: str = 'rcn'):
        super().__init__(output_key_to_shape, targets)
        self.source_name = source_name
        self.force_cpu = force_cpu
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.should_norm_hidden = should_norm_hidden
        self.hidden = None
        self.linear = None
        self.norm_layers = None
        self.apply_at_most_one_data_id = apply_at_most_one_data_id
        self.should_norm = should_norm
        self.activation_fn = activation_fn
        self.penultimate_reconstruction_penalty = HiddenReconstructionPenalty(
            penultimate_reconstruction_penalty_coefficient,
            penultimate_reconstruction_l1_weight_coefficient,
            penultimate_reconstruction_output_name,
            hidden_activation,
            should_norm_hidden)

    def _instantiate(self, name_to_num_channels):
        in_channels = name_to_num_channels[self.source_name]
        if self.hidden_sizes is not None:
            hidden_sizes = [self.hidden_sizes] if np.ndim(self.hidden_sizes) == 0 else self.hidden_sizes
            hidden_modules = list()
            for index_hidden in range(len(hidden_sizes)):
                current_in = in_channels if index_hidden == 0 else hidden_sizes[index_hidden - 1]
                hidden_modules.append(
                    LinearWithLayerNorm(
                        current_in,
                        hidden_sizes[index_hidden], self.hidden_activation, should_norm=self.should_norm_hidden))
            self.hidden = torch.nn.Sequential(*hidden_modules)
            in_channels = hidden_sizes[-1]
        self.linear = nn.Linear(in_channels, sum(self.splits))
        result = OrderedDict()
        for key in self.output_key_to_shape:
            result[key] = int(np.prod(self.output_key_to_shape[key]))
        if self.should_norm:
            self.norm_layers = torch.nn.ModuleList(
                modules=list(torch.nn.LayerNorm(result[k]) for k in result))
        for key in name_to_num_channels:
            if isinstance(key, tuple) and key[0] == self.source_name:
                for result_key in self.output_key_to_shape:
                    result[(result_key,) + key[1:]] = name_to_num_channels[key]
        self.penultimate_reconstruction_penalty.instantiate(sum(self.splits), in_channels)
        return result

    def forward(self, batch):
        x = batch[self.source_name]
        if self.hidden is not None:
            x = self.hidden(x)
        if self.force_cpu:
            x = x.cpu()
        predictions = self.linear(x)
        predictions = torch.split(predictions, self.splits, dim=-1)
        if self.activation_fn is not None:
            predictions = [self.activation_fn(p) for p in predictions]
        if self.should_norm:
            predictions = [norm(p) for norm, p in zip(self.norm_layers, predictions)]
        result = OrderedDict()
        result.update(self.penultimate_reconstruction_penalty(torch.cat(predictions, dim=-1), x))
        assert(len(self.output_key_to_shape) == len(predictions))
        for k, p in zip(self.output_key_to_shape, predictions):
            for key in batch:
                if isinstance(key, tuple) and key[0] == self.source_name:
                    result[(k,) + key[1:]] = batch[key]
            p = p.view(p.size()[:-1] + self.output_key_to_shape[k])
            result[k] = p

            if isinstance(self.apply_at_most_one_data_id, dict):
                apply_at_most_one_data_id = self.apply_at_most_one_data_id[k] \
                    if k in self.apply_at_most_one_data_id else False
            else:
                apply_at_most_one_data_id = self.apply_at_most_one_data_id

            if (apply_at_most_one_data_id == 'if_no_target' and k not in batch and (k, 'data_ids') in batch) \
                    or apply_at_most_one_data_id is True:
                data_ids = at_most_one_data_id(batch[(k, 'data_ids')])
                indicator_valid = data_ids >= 0
                result[k] = result[k][indicator_valid]
                result[(k, 'data_ids')] = data_ids[indicator_valid]
                result[(k, 'example_ids')] = torch.arange(len(data_ids), device=data_ids.device)[indicator_valid]

        return result

    def compute_penalties(self, batch, predictions, loss_dict):
        return self.penultimate_reconstruction_penalty.compute_penalties(batch, predictions, loss_dict)


class KeyedQuasiAttention(KeyedBase):

    def __init__(
            self,
            source_name: Union[str, Tuple[str, ...]],
            hidden_sizes: Optional[Union[int, Sequence[int]]] = None,
            hidden_activation: Optional[Callable[[Tensor], Tensor]] = gelu,
            should_norm_hidden: bool = True,
            force_cpu: bool = False,
            output_key_to_shape: Optional[Mapping[str, Union[int, Tuple[int, ...]]]] = None,
            targets: Optional[Union[str, Iterable[str]]] = None,
            apply_at_most_one_data_id: Union[str, bool, Mapping[str, Union[str, bool]]] = False,
            should_norm: bool = False,
            activation_fn: Optional[Callable[[Tensor], Tensor]] = None,
            penultimate_reconstruction_penalty_coefficient: float = 0,
            penultimate_reconstruction_l1_weight_coefficient: float = 0,
            penultimate_reconstruction_output_name: str = 'rcn'):
        super().__init__(output_key_to_shape, targets)
        self.source_name = source_name
        self.force_cpu = force_cpu
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.should_norm_hidden = should_norm_hidden
        self.hidden = None
        self.quasi_attention = None
        self.norm_layers = None
        self.apply_at_most_one_data_id = apply_at_most_one_data_id
        self.should_norm = should_norm
        self.activation_fn = activation_fn
        self.penultimate_reconstruction_penalty = HiddenReconstructionPenalty(
            penultimate_reconstruction_penalty_coefficient,
            penultimate_reconstruction_l1_weight_coefficient,
            penultimate_reconstruction_output_name,
            hidden_activation,
            should_norm_hidden)

    def _instantiate(self, name_to_num_channels):
        in_channels = name_to_num_channels[self.source_name]
        if self.hidden_sizes is not None:
            hidden_sizes = [self.hidden_sizes] if np.ndim(self.hidden_sizes) == 0 else self.hidden_sizes
            hidden_modules = list()
            for index_hidden in range(len(hidden_sizes)):
                current_in = in_channels if index_hidden == 0 else hidden_sizes[index_hidden - 1]
                hidden_modules.append(
                    QuasiAttention(
                        current_in,
                        hidden_sizes[index_hidden],
                        activation_function=self.hidden_activation,
                        should_norm=self.should_norm_hidden))
            self.hidden = torch.nn.Sequential(*hidden_modules)
            in_channels = hidden_sizes[-1]
        self.quasi_attention = QuasiAttention(
            in_channels, sum(self.splits), activation_function=None, should_norm=False)
        result = OrderedDict()
        for key in self.output_key_to_shape:
            result[key] = int(np.prod(self.output_key_to_shape[key]))
        if self.should_norm:
            self.norm_layers = torch.nn.ModuleList(
                modules=list(torch.nn.LayerNorm(result[k]) for k in result))
        for key in name_to_num_channels:
            if isinstance(key, tuple) and key[0] == self.source_name:
                for result_key in self.output_key_to_shape:
                    result[(result_key,) + key[1:]] = name_to_num_channels[key]
        self.penultimate_reconstruction_penalty.instantiate(sum(self.splits), in_channels)
        return result

    def forward(self, batch):
        x = batch[self.source_name]
        if self.hidden is not None:
            x = self.hidden(x)
        if self.force_cpu:
            x = x.cpu()
        predictions = self.quasi_attention(x)
        predictions = torch.split(predictions, self.splits, dim=-1)
        if self.activation_fn is not None:
            predictions = [self.activation_fn(p) for p in predictions]
        if self.should_norm:
            predictions = [norm(p) for norm, p in zip(self.norm_layers, predictions)]
        result = OrderedDict()
        result.update(self.penultimate_reconstruction_penalty(torch.cat(predictions, dim=-1), x))
        assert(len(self.output_key_to_shape) == len(predictions))
        for k, p in zip(self.output_key_to_shape, predictions):
            for key in batch:
                if isinstance(key, tuple) and key[0] == self.source_name:
                    result[(k,) + key[1:]] = batch[key]
            p = p.view(p.size()[:-1] + self.output_key_to_shape[k])
            result[k] = p

            if isinstance(self.apply_at_most_one_data_id, dict):
                apply_at_most_one_data_id = self.apply_at_most_one_data_id[k] \
                    if k in self.apply_at_most_one_data_id else False
            else:
                apply_at_most_one_data_id = self.apply_at_most_one_data_id

            if (apply_at_most_one_data_id == 'if_no_target' and k not in batch and (k, 'data_ids') in batch) \
                    or apply_at_most_one_data_id is True:
                data_ids = at_most_one_data_id(batch[(k, 'data_ids')])
                indicator_valid = data_ids >= 0
                result[k] = result[k][indicator_valid]
                result[(k, 'data_ids')] = data_ids[indicator_valid]
                result[(k, 'example_ids')] = torch.arange(len(data_ids), device=data_ids.device)[indicator_valid]

        return result

    def compute_penalties(self, batch, predictions, loss_dict):
        return self.penultimate_reconstruction_penalty.compute_penalties(batch, predictions, loss_dict)


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
                    # noinspection PyUnresolvedReferences
                    self.conv_hidden.append(torch.nn.Linear(in_sequence_channels, self.conv_hidden_channels))
                else:
                    # noinspection PyUnresolvedReferences
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
            # noinspection PyUnresolvedReferences
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


class KeyedSingleTargetSpanMaxPool(KeyedBase):

    def __init__(
            self,
            num_spans,
            sequence_source_name,
            span_source_name,
            output_key_to_shape=None,
            targets=None):
        super().__init__(output_key_to_shape, targets)
        self.num_spans = num_spans
        self.sequence_source_name = sequence_source_name
        self.span_source_name = span_source_name
        self.linear = None
        self.named_span_encoder = NamedSpanEncoder(list(range(num_spans)))

    def _instantiate(self, name_to_num_channels):
        in_sequence_channels = name_to_num_channels[self.sequence_source_name]
        self.linear = torch.nn.Linear(in_sequence_channels * self.num_spans, sum(self.splits))

        result = OrderedDict(self.output_key_to_shape)
        for key in name_to_num_channels:
            if isinstance(key, tuple) and key[0] == self.sequence_source_name:
                for result_key in self.output_key_to_shape:
                    result[(result_key,) + key[1:]] = name_to_num_channels[key]
        return result

    def forward(self, batch):
        if self.span_source_name not in batch:
            return OrderedDict()
        span_ids = batch[self.span_source_name]
        span_indicators, is_fully_decoded = self.named_span_encoder.torch_span_indicators(span_ids)
        if not is_fully_decoded:
            # this has more spans than this module can handle
            return OrderedDict()
        span_embeddings = list()
        for index_span, span_name in enumerate(span_indicators):
            span_indicator = torch.unsqueeze(span_indicators[span_name], dim=2)
            span_values = torch.full_like(batch[self.sequence_source_name], -float('inf'))
            span_values.masked_scatter_(
                span_indicator, torch.masked_select(batch[self.sequence_source_name], span_indicator))
            max_values, _ = torch.max(span_values, dim=1)
            span_embeddings.append(max_values)
        prediction_input = torch.cat(span_embeddings, dim=1)
        predictions = self.linear(prediction_input)
        predictions = torch.split(predictions, self.splits, dim=-1)
        assert (len(self.output_key_to_shape) == len(predictions))
        result = OrderedDict()
        for k, p in zip(self.output_key_to_shape, predictions):
            p = p.view(p.size()[:1] + self.output_key_to_shape[k])
            result[k] = p
        return result


class KeyedConcat(GraphPart):

    def __init__(
            self,
            source_names: Iterable[Union[str, Tuple[str, ...]]],
            output_name: str,
            activation_fn: Optional[Callable[[Tensor], Tensor]] = None):
        super().__init__()
        self.source_names = OrderedDict((k, None) for k in source_names)
        self.output_name = output_name
        self.activation_fn = activation_fn

    def resolve_placeholders(self, placeholder_name_to_fields, field_shapes, num_response_data_fields):
        source_names = OrderedDict()
        for source in self.source_names:
            if source in placeholder_name_to_fields:
                for field in placeholder_name_to_fields[source]:
                    if field not in self.source_names:
                        source_names[field] = None
            else:
                source_names[source] = None
        self.source_names = source_names

    def instantiate(self, name_to_num_channels):
        num_channels = 0
        source_names = OrderedDict()
        for source in self.source_names:
            source_names[source] = name_to_num_channels[source]
            num_channels += int(np.prod(name_to_num_channels[source]))
        self.source_names = source_names
        result = OrderedDict()
        result[self.output_name] = num_channels
        return result

    def forward(self, batch):
        result = list()
        for source in self.source_names:
            num_channels = self.source_names[source]
            if np.ndim(num_channels) > 0:
                result.append(
                    batch[source].view(batch[source].size()[:-len(num_channels)] + (int(np.prod(num_channels)),)))
            elif num_channels == 1 and len(batch[source].size()) == 1:
                result.append(batch[source].view(batch[source].size() + (1,)))
            else:
                result.append(batch[source])
        output = OrderedDict()
        output[self.output_name] = torch.cat(result, dim=-1)
        if self.activation_fn is not None:
            output[self.output_name] = self.activation_fn(output[self.output_name])
        return output
