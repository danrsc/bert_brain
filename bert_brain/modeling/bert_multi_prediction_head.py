import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn
from transformers import BertPreTrainedModel, BertModel

from .multi_head_config import BertMultiPredictionHeadConfig

__all__ = ['MultiPredictionHead', 'BertMultiPredictionHead', 'BertOutputSupplement', 'LazyBertOutputBatch',
           'LazyBertOutputNumChannels']


logger = logging.getLogger(__name__)


class MultiPredictionHead(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            num_in_layers,
            dropout_prob,
            head_graph_parts,
            token_supplemental_key_to_shape=None,
            token_supplemental_skip_dropout_keys=None,
            pooled_supplemental_key_to_shape=None,
            pooled_supplemental_skip_dropout_keys=None):

        super().__init__()

        self.dropout = torch.nn.Dropout(dropout_prob)

        self.token_supplement = None
        if token_supplemental_key_to_shape is not None and len(token_supplemental_key_to_shape) > 0:
            self.token_supplement = BertOutputSupplement(
                in_channels,
                supplemental_dropout_prob=dropout_prob,
                is_sequence_supplement=True,
                supplement_key_to_shape=token_supplemental_key_to_shape,
                skip_dropout_keys=token_supplemental_skip_dropout_keys)
        self.pooled_supplement = None
        if pooled_supplemental_key_to_shape is not None and len(pooled_supplemental_key_to_shape) > 0:
            self.pooled_supplement = BertOutputSupplement(
                in_channels,
                supplemental_dropout_prob=dropout_prob,
                is_sequence_supplement=False,
                supplement_key_to_shape=pooled_supplemental_key_to_shape,
                skip_dropout_keys=pooled_supplemental_skip_dropout_keys)

        name_to_num_channels = LazyBertOutputNumChannels(
            self.token_supplement, self.pooled_supplement, in_channels, num_in_layers)
        for key in head_graph_parts:
            graph_part_num_channels = head_graph_parts[key].instantiate(name_to_num_channels.copy())
            for k in graph_part_num_channels:
                if k in name_to_num_channels:
                    raise ValueError('Duplicate output: {}'.format(k))
                name_to_num_channels[k] = graph_part_num_channels[k]

        # noinspection PyTypeChecker
        self.head_graph_parts = torch.nn.ModuleDict(modules=[(k, head_graph_parts[k]) for k in head_graph_parts])

    def forward(self, sequence_output, pooled_output, batch, dataset):
        batch_inputs = LazyBertOutputBatch(
            sequence_output,
            pooled_output,
            batch,
            self.dropout,
            self.token_supplement,
            self.pooled_supplement)

        outputs = OrderedDict()
        # noinspection PyTypeChecker
        for name in self.head_graph_parts:
            # noinspection PyUnresolvedReferences
            head = self.head_graph_parts[name]
            head_outputs = head(batch_inputs)

            for k in head_outputs:
                if k in outputs:  # don't allow two graph parts to output the same key
                    raise ValueError('multiple predictions made for key: {}'.format(k))
                else:
                    outputs[k] = head_outputs[k]
                    # overwrite batch_inputs[k] if it exists
                    batch_inputs[k] = head_outputs[k]

        # pass through anything that was not output by a graph part
        for k in batch:
            if k not in outputs:
                outputs[k] = batch[k]

        # set the outputs as the batch, which now includes passed through keys
        batch = outputs

        return batch


class BertOutputSupplement(torch.nn.Module):

    def __init__(
            self, in_channels, supplemental_dropout_prob, is_sequence_supplement, supplement_key_to_shape,
            skip_dropout_keys=None):
        super().__init__()
        self.is_sequence_supplement = is_sequence_supplement
        self.in_channels = in_channels
        self.dropout = torch.nn.Dropout(supplemental_dropout_prob)
        self.supplement_key_to_shape = OrderedDict()
        if supplement_key_to_shape is not None:
            self.supplement_key_to_shape.update(supplement_key_to_shape)
        self.skip_dropout_keys = set()
        if skip_dropout_keys is not None:
            self.skip_dropout_keys.update(skip_dropout_keys)

    def supplement_channels(self):
        return sum(int(np.prod(self.supplement_key_to_shape[k])) for k in self.supplement_key_to_shape)

    def out_channels(self):
        return self.in_channels + self.supplement_channels()

    def forward(self, x, batch):
        # we expect that dropout has already been applied to sequence_output / pooled_output
        all_values = [x]
        for key in self.supplement_key_to_shape:
            values = batch[key]
            shape_part = values.size()[:2] if self.is_sequence_supplement else values.size()[:1]
            values = values.view(
                shape_part + (int(np.prod(self.supplement_key_to_shape[key])),)).type(all_values[0].dtype)
            if key not in self.skip_dropout_keys:
                values = self.dropout(values)
            all_values.append(values)
        return torch.cat(all_values, dim=2 if self.is_sequence_supplement else 1)


class BertMultiPredictionHead(BertPreTrainedModel):

    config_class = BertMultiPredictionHeadConfig

    def __init__(self, config: BertMultiPredictionHeadConfig):
        config.output_hidden_states = True
        super(BertMultiPredictionHead, self).__init__(config)
        self.bert = BertModel(config)
        # noinspection PyUnresolvedReferences
        self.prediction_head = MultiPredictionHead(
            config.hidden_size,
            len(self.bert.encoder.layer),
            config.hidden_dropout_prob,
            config.head_graph_parts,
            config.token_supplemental_key_to_shape,
            config.token_supplemental_skip_dropout_keys,
            config.pooled_supplemental_key_to_shape,
            config.pooled_supplemental_skip_dropout_keys)
        self.init_weights()

    def forward(self, batch, dataset):
        bert_outputs = self.bert(
            batch['token_ids'],
            token_type_ids=batch['type_ids'] if 'type_ids' in batch else None,
            attention_mask=batch['mask'] if 'mask' in batch else None)
        pooled_output = bert_outputs[1]
        sequence_output = bert_outputs[2]
        # noinspection PyCallingNonCallable
        return self.prediction_head(sequence_output, pooled_output, batch, dataset)

    def to(self, *args, **kwargs):

        # noinspection PyProtectedMember, PyUnresolvedReferences
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)

        if dtype is not None:
            if not dtype.is_floating_point:
                raise TypeError('nn.Module.to only accepts floating point '
                                'dtypes, but got desired dtype={}'.format(dtype))

        forced_cpu = list()

        def set_forced_cpu(module):
            for child in module.children():
                set_forced_cpu(child)
            force_cpu = getattr(module, 'force_cpu', False)
            if force_cpu:
                def set_forced_cpu_tensor(t):
                    forced_cpu.append(t)
                    return t

                # noinspection PyProtectedMember
                module._apply(set_forced_cpu_tensor)

        set_forced_cpu(self)

        def is_forced_cpu(t):
            for have in forced_cpu:
                if have.is_set_to(t):
                    return True
            return False

        def convert(t):
            if is_forced_cpu(t):
                return t.to(torch.device('cpu'), dtype if t.is_floating_point() else None, non_blocking)
            else:
                return t.to(device, dtype if t.is_floating_point() else None, non_blocking)

        self._apply(convert)

    def cuda(self, device=None):
        forced_cpu = list()

        def set_forced_cpu(module):
            for child in module.children():
                set_forced_cpu(child)
            force_cpu = getattr(module, 'force_cpu', False)
            if force_cpu:
                def set_forced_cpu_tensor(t):
                    forced_cpu.append(t)
                    return t

                # noinspection PyProtectedMember
                module._apply(set_forced_cpu_tensor)

        set_forced_cpu(self)

        def is_forced_cpu(t):
            for have in forced_cpu:
                if have.is_set_to(t):
                    return True
            return False

        def convert(t):
            if is_forced_cpu(t):
                return t.cpu()
            return t.cuda(device)
        return self._apply(convert)


class LazyBertOutputNumChannels:

    def __init__(self, sequence_supplement, pooled_supplement, in_channels, num_layers):
        self.sequence_supplement = sequence_supplement
        self.pooled_supplement = pooled_supplement
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.name_to_num_channels = OrderedDict()

    def copy(self):
        result = LazyBertOutputNumChannels(
            self.sequence_supplement, self.pooled_supplement, self.in_channels, self.num_layers)
        result.name_to_num_channels = OrderedDict(self.name_to_num_channels)
        return result

    def __delitem__(self, key):
        del self.name_to_num_channels[key]

    def __iter__(self):
        return iter(self.name_to_num_channels)

    def __len__(self):
        return len(self.name_to_num_channels)

    def __setitem__(self, key, value):
        self.name_to_num_channels[key] = value

    def __getitem__(self, item):
        if item in self.name_to_num_channels:
            return self.name_to_num_channels[item]

        if isinstance(item, (tuple, list)):
            if item[0] == 'bert':
                kind = item[1]
                layer = item[2] if len(item) > 2 else -1
                if kind == 'sequence' or kind == 'untransformed_pooled':
                    if layer == 'all':
                        num_layers = self.num_layers
                    elif np.ndim(layer) == 0:
                        num_layers = 1
                    else:
                        num_layers = len(layer)
                    if kind == 'untransformed_pooled':
                        supplement = self.pooled_supplement
                    else:
                        supplement = self.sequence_supplement
                elif kind == 'pooled':
                    if np.ndim(layer) != 0 or not isinstance(layer, int):
                        raise KeyError('Cannot get pooled result from a layer other than the last layer. '
                                       'Requested layer: {}. Did you mean untransformed_pooled?'.format(layer))
                    if layer < 0:
                        layer += self.num_layers
                    if layer != self.num_layers - 1:
                        raise KeyError('Cannot get pooled result from a layer other than the last layer. '
                                       'Requested layer: {}. Did you mean untransformed_pooled?'.format(item[2]))
                    num_layers = 1
                    supplement = self.pooled_supplement
                else:
                    raise KeyError('Unrecognized kind: {}'.format(kind))

                self.name_to_num_channels[item] = self.in_channels * num_layers + (
                    supplement.supplement_channels() if supplement is not None else 0)

        return self.name_to_num_channels[item]


class LazyBertOutputBatch:

    def __init__(
            self,
            sequence_output,
            pooled_output,
            batch,
            dropout_layer,
            sequence_supplement,
            pooled_supplement):
        self.pooled_output = pooled_output
        self.sequence_output = sequence_output
        self.batch = OrderedDict(batch)
        self._dropped_out = dict()
        self.dropout_layer = dropout_layer
        self.sequence_supplement = sequence_supplement
        self.pooled_supplement = pooled_supplement

    def __delitem__(self, key):
        del self.batch[key]

    def __iter__(self):
        return iter(self.batch)

    def __len__(self):
        return len(self.batch)

    def __setitem__(self, key, value):
        self.batch[key] = value

    def __contains__(self, item):
        try:
            _ = self[item]
            return True
        except KeyError:
            return False

    def __getitem__(self, item):
        if item in self.batch:
            return self.batch[item]

        if isinstance(item, (tuple, list)):
            if item[0] == 'bert':
                kind = item[1]
                layer = item[2] if len(item) > 2 else -1
                if kind == 'sequence' or kind == 'untransformed_pooled':
                    if layer == 'all':
                        indices = range(len(self.sequence_output))
                    elif np.ndim(layer) == 0:
                        indices = [layer]
                    else:
                        indices = layer
                    x = list()
                    for layer in indices:
                        if layer < 0:
                            layer += len(self.sequence_output)
                            if layer < 0 or layer >= len(self.sequence_output):
                                raise KeyError('Invalid layer requested: {}'.format(item[2]))
                        if ('sequence', layer) not in self._dropped_out:
                            self._dropped_out[('sequence', layer)] = self.dropout_layer(self.sequence_output[layer])
                        x.append(self._dropped_out[('sequence', layer)])
                    if kind == 'untransformed_pooled':
                        x = [x_[:, 0] for x_ in x]
                        supplement = self.pooled_supplement
                    else:
                        supplement = self.sequence_supplement
                elif kind == 'pooled':
                    if np.ndim(layer) != 0 or not isinstance(layer, int):
                        raise KeyError('Cannot get pooled result from a layer other than the last layer. '
                                       'Requested layer: {}. Did you mean untransformed_pooled?'.format(layer))
                    if layer < 0:
                        layer += len(self.sequence_output)
                    if layer != len(self.sequence_output) - 1:
                        raise KeyError('Cannot get pooled result from a layer other than the last layer. '
                                       'Requested layer: {}. Did you mean untransformed_pooled?'.format(item[2]))
                    if (kind, layer) not in self._dropped_out:
                        self._dropped_out[(kind, layer)] = self.dropout_layer(self.pooled_output)
                    x = [self._dropped_out[(kind, layer)]]
                    supplement = self.pooled_supplement
                else:
                    raise KeyError('Unrecognized kind: {}'.format(kind))

                if len(x) == 1:
                    x = x[0]
                else:
                    x = torch.cat(x, dim=-1)
                if supplement is not None:
                    self.batch[item] = supplement(x, self.batch)
                else:
                    self.batch[item] = x

        return self.batch[item]
