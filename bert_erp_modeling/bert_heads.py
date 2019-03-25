from collections import OrderedDict
import logging

import numpy as np
import torch
from torch import nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel

from bert_erp_modeling.utility_modules import GroupPool, at_most_one_data_id

logger = logging.getLogger(__name__)


__all__ = ['BertMultiPredictionHead', 'KeyedLinear', 'KeyedGroupPooledLinear', 'MultiPredictionHead',
           'BertOutputSupplement']


class KeyedLinear(torch.nn.Module):

    def __init__(self, is_sequence, in_sequence_channels, in_pooled_channels, prediction_key_to_shape):
        super(KeyedLinear, self).__init__()
        self.is_sequence = is_sequence
        self.prediction_key_to_shape = OrderedDict(prediction_key_to_shape)
        self.splits = [int(np.prod(self.prediction_key_to_shape[k])) for k in self.prediction_key_to_shape]
        self.linear = nn.Linear(in_sequence_channels if self.is_sequence else in_pooled_channels, sum(self.splits))

    def forward(self, sequence_output, pooled_output, batch):
        x = sequence_output if self.is_sequence else pooled_output
        predictions = self.linear(x)
        predictions = torch.split(predictions, self.splits, dim=-1)
        result = OrderedDict()
        assert(len(self.prediction_key_to_shape) == len(predictions))
        for k, p in zip(self.prediction_key_to_shape, predictions):
            if self.is_sequence:
                p = p.view(p.size()[:2] + self.prediction_key_to_shape[k])
            else:
                p = p.view(p.size()[:1] + self.prediction_key_to_shape[k])
            result[k] = p
            if not self.is_sequence and k not in batch:
                # we are in data_ids mode, there must be at most one valid data_id per example
                data_ids = at_most_one_data_id(batch[(k, 'data_ids')])
                indicator_valid = data_ids >= 0
                result[k] = result[k][indicator_valid]
                result[(k, 'data_ids')] = data_ids[indicator_valid]
        return result


class KeyedGroupPooledLinear(torch.nn.Module):

    def __init__(self, in_sequence_channels, in_pooled_channels, prediction_key_to_shape):
        super().__init__()
        self.group_pool = GroupPool()
        self.linear = KeyedLinear(
            is_sequence=False, in_channels=in_sequence_channels, prediction_key_to_shape=prediction_key_to_shape)

    def forward(self, sequence_output, pooled_output, batch):
        all_data_ids = [batch[(k, 'data_ids')] for k in self.prediction_key_to_shape]
        for idx in range(1, len(all_data_ids)):
            if not torch.equal(all_data_ids[0], all_data_ids[idx]):
                raise ValueError('Inconsistent data_ids cannot be used within the same instance of FMRIHead')
        data_ids = all_data_ids[0]
        pooled, groups, example_ids = self.group_pool(sequence_output, data_ids)
        result = self.linear(None, pooled, batch)
        keys = [k for k in result]
        for k in keys:
            result[(k, 'data_ids')] = groups
            result[(k, 'example_ids')] = example_ids
        return result


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


class MultiPredictionHead(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            supplemental_dropout_prob,
            prediction_head_settings,
            token_supplemental_key_to_shape=None,
            token_supplemental_skip_dropout_keys=None,
            pooled_supplemental_key_to_shape=None,
            pooled_supplemental_skip_dropout_keys=None):
        super().__init__()

        in_sequence_channels = in_channels
        in_pooled_channels = in_channels
        self.token_supplement = None
        if token_supplemental_key_to_shape is not None and len(token_supplemental_key_to_shape) > 0:
            self.token_supplement = BertOutputSupplement(
                in_channels,
                supplemental_dropout_prob=supplemental_dropout_prob,
                is_sequence_supplement=True,
                supplement_key_to_shape=token_supplemental_key_to_shape,
                skip_dropout_keys=token_supplemental_skip_dropout_keys)
            in_sequence_channels = self.token_supplement.out_channels()
        self.pooled_supplement = None
        if pooled_supplemental_key_to_shape is not None and len(pooled_supplemental_key_to_shape) > 0:
            self.pooled_supplement = BertOutputSupplement(
                in_channels,
                supplemental_dropout_prob=supplemental_dropout_prob,
                is_sequence_supplement=False,
                supplement_key_to_shape=pooled_supplemental_key_to_shape,
                skip_dropout_keys=pooled_supplemental_skip_dropout_keys)
            in_pooled_channels = self.pooled_supplement.out_channels()

        self.prediction_heads = torch.nn.ModuleList(modules=[
            ph[0].head_type(
                in_sequence_channels=in_sequence_channels,
                in_pooled_channels=in_pooled_channels,
                prediction_key_to_shape=ph[1],
                **ph[0].kwargs)
            for ph in prediction_head_settings])

    def forward(self, sequence_output, pooled_output, batch, dataset):
        if self.token_supplement is not None:
            sequence_output = self.token_supplement(sequence_output, batch)
        if self.pooled_supplement is not None:
            pooled_output = self.pooled_supplement(pooled_output, batch)

        predictions = OrderedDict()
        for head in self.prediction_heads:
            head_predictions = head(sequence_output, pooled_output, batch)
            for k in head_predictions:
                if k in predictions:
                    raise ValueError('multiple predictions made for key: {}'.format(k))
                else:
                    predictions[k] = head_predictions[k]

        # fetch the data that was too expensive to put in batch as padded
        for k in predictions:
            if isinstance(k, tuple) and len(k) == 2 and k[1] == 'data_ids':
                group_data = dataset.get_data_for_data_ids(k[0], predictions[k].cpu().numpy())
                batch[k[0]] = group_data.to(predictions[k].device)

        return predictions


class BertMultiPredictionHead(BertPreTrainedModel):

    def __init__(
            self,
            config,
            prediction_head_settings,
            token_supplemental_key_to_shape=None,
            token_supplemental_skip_dropout_keys=None,
            pooled_supplemental_key_to_shape=None,
            pooled_supplemental_skip_dropout_keys=None):

        super(BertMultiPredictionHead, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.prediction_head = MultiPredictionHead(
            config.hidden_size,
            config.hidden_dropout_prob,
            prediction_head_settings,
            token_supplemental_key_to_shape,
            token_supplemental_skip_dropout_keys,
            pooled_supplemental_key_to_shape,
            pooled_supplemental_skip_dropout_keys)
        self.apply(self.init_bert_weights)

    def forward(self, batch, dataset):
        sequence_output, pooled_output = self.bert(
            batch['token_ids'],
            token_type_ids=batch['type_ids'] if 'type_ids' in batch else None,
            attention_mask=batch['mask'] if 'mask' in batch else None,
            output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        return self.prediction_head(sequence_output, pooled_output, batch, dataset)
