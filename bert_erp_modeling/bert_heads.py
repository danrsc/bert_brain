from collections import OrderedDict
import logging

import numpy as np
import torch
from torch import nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel

logger = logging.getLogger(__name__)


__all__ = ['BertMultiHead']


class KeyedLinear(torch.nn.Module):

    def __init__(self, is_sequence, input_size, prediction_key_to_shape):
        super(KeyedLinear, self).__init__()
        self.is_sequence = is_sequence
        self.prediction_key_to_shape = OrderedDict(prediction_key_to_shape)
        self.splits = [int(np.prod(self.prediction_key_to_shape[k])) for k in self.prediction_key_to_shape]
        self.linear = nn.Linear(input_size, sum(self.splits))

    def forward(self, x):
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
        return result


class GroupPool(torch.nn.Module):

    def forward(self, x, groupby):

        # first attach an example_id to the groups to ensure that we don't pool across examples in the batch

        # array of shape (batch, sequence, 1) which gives identifies which example
        example_ids = torch.arange(groupby.size()[0]).view((groupby.size()[0], 1, 1)).repeat(1, groupby.size()[1], 1)
        # -> (batch, sequence, 2): attach example_id to each group
        groupby = torch.cat((groupby.view(groupby.size() + (1,)), example_ids), dim=2)

        # -> (batch * sequence, 2)
        groupby = groupby.view((groupby.size()[0] * groupby.size()[1], groupby.size()[2]))

        # each group is a (group, example_id) tuple
        groups, group_indices = torch.unique(groupby, return_inverse=True, dim=0)

        # -> (batch, sequence)
        group_indices = group_indices.view((x.size()[0], x.size()[1]))

        # remove the example_id
        groups = groups[:, 0]

        pooled = torch.zeros((groups.size()[0],) + x.size()[1:], dtype=x.dtype)
        counts = torch.zeros_like(pooled, dtype=x.dtype)
        pooled.scatter_add_(dim=0, index=group_indices, other=x)
        counts.scatter_add_(dim=0, index=group_indices, other=torch.ones_like(group_indices))
        pooled = pooled / counts

        return pooled, groups


class BertMultiHead(BertPreTrainedModel):

    @staticmethod
    def _setup_linear_layer(hidden_size, prediction_key_to_shape, input_key_to_shape, which):
        linear = None
        if prediction_key_to_shape is not None and len(prediction_key_to_shape) > 0:
            if input_key_to_shape is not None:
                input_key_to_shape = OrderedDict(input_key_to_shape)
                input_count = sum(int(np.prod(input_key_to_shape[k])) for k in input_key_to_shape)
            else:
                input_count = 0

            logger.info('{} head active'.format(which))
            linear = KeyedLinear(which == 'token', hidden_size + input_count, prediction_key_to_shape)
        else:
            logger.info('{} head inactive'.format(which))
            if input_key_to_shape is not None and len(input_key_to_shape) > 0:
                logger.warning('input_key_to_shape is not None, but {} prediction is inactive'.format(which))
        return input_key_to_shape, linear

    def __init__(
            self,
            config,
            token_prediction_key_to_shape=None,
            token_level_input_key_to_shape=None,
            pooled_prediction_key_to_shape=None,
            pooled_input_key_to_shape=None,
            group_pooled_prediction_key_to_shape=None,
            group_pooled_input_key_to_shape=None):

        has_prediction = False
        for p_to_s in (
                token_prediction_key_to_shape, pooled_prediction_key_to_shape, group_pooled_prediction_key_to_shape):
            if p_to_s is not None and len(p_to_s) > 0:
                has_prediction = True
                break

        if not has_prediction:
            raise ValueError('At least one of token_prediction_key_to_shape, pooled_prediction_key_to_shape, '
                             'and group_pooled_prediction_key_to_shape must be non-empty')

        super(BertMultiHead, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.token_level_input_key_to_shape, self.token_linear = BertMultiHead._setup_linear_layer(
            config.hidden_size, token_prediction_key_to_shape, token_level_input_key_to_shape, 'token')
        self.pooled_input_key_to_shape, self.pooled_linear = BertMultiHead._setup_linear_layer(
            config.hidden_size, pooled_prediction_key_to_shape, pooled_input_key_to_shape, 'pooled')
        self.group_pooled_input_key_to_shape, self.group_pooled_linear = BertMultiHead._setup_linear_layer(
            config.hidden_size, group_pooled_prediction_key_to_shape, group_pooled_input_key_to_shape, 'group_pooled')
        self.group_pool = None
        if self.group_pooled_linear is not None:
            self.group_pool = GroupPool()
        self.apply(self.init_bert_weights)

    def forward(self, batch):
        sequence_output, pooled_output = self.bert(
            batch['input_ids'],
            token_type_ids=batch['input_type_ids'] if 'input_type_ids' in batch else None,
            attention_mask=batch['input_mask'] if 'input_mask' in batch else None,
            output_all_encoded_layers=False)

        if self.token_linear is not None or self.group_pooled_linear is not None:
            sequence_output = self.dropout(sequence_output)
        if self.pooled_linear is not None:
            pooled_output = self.dropout(pooled_output)

        token_input = sequence_output
        if self.token_level_input_key_to_shape is not None:
            all_values = [sequence_output]
            for key in self.token_level_input_key_to_shape:
                values = batch[key]
                values = values.view(
                    values.size()[:2] + (int(np.prod(self.token_level_input_key_to_shape[key])),))
                all_values.append(self.dropout(values.type(sequence_output.dtype)))
            token_input = torch.cat(all_values, dim=2)

        pooled_input = pooled_output
        if self.pooled_input_key_to_shape is not None:
            all_values = [pooled_output]
            for key in self.pooled_input_key_to_shape:
                values = batch[key]
                values = values.view(values.size()[:1] + (int(np.prod(self.pooled_input_key_to_shape[key])),))
                all_values.append(self.dropout(values.type(pooled_output.dtype)))
            pooled_input = torch.cat(all_values, dim=1)

        group_pool_predictions = None
        groups = None
        if self.group_pool is not None:
            grouped_input = sequence_output
            if self.group_pooled_input_key_to_shape is not None:
                all_values = [sequence_output]
                for key in self.group_pooled_input_key_to_shape:
                    values = batch[key]
                    values = values.view(values.size()[:2] + (int(np.prod(self.group_pooled_input_key_to_shape[key])),))
                    all_values.append(self.dropout(values.type(sequence_output.dtype)))
                grouped_input = torch.cat(all_values, dim=2)

            grouped_input, groups = self.group_pool(grouped_input, batch['data_ids'])
            group_pool_predictions = self.group_pooled_linear(grouped_input)

        token_predictions = self.token_linear(token_input) if self.token_linear is not None else None
        pooled_predictions = self.pooled_linear(pooled_input) if self.pooled_linear is not None else None

        predictions = OrderedDict()
        if token_predictions is not None:
            for k in token_predictions:
                if k in predictions:
                    raise ValueError('multiple predictions made for key: {}'.format(k))
                predictions[k] = token_predictions[k]
        if pooled_predictions is not None:
            for k in pooled_predictions:
                if k in predictions:  # this key is also in token_predictions; add the two
                    predictions[k] = predictions[k] + pooled_predictions[k].unsqueeze(dim=1)
                else:
                    predictions[k] = pooled_predictions[k]
        if group_pool_predictions is not None:
            for k in group_pool_predictions:
                if k in predictions:
                    raise ValueError('multiple predictions made for key: {}'.format(k))
                predictions[k] = group_pool_predictions[k]
            predictions['group_pool_groups'] = groups

        return predictions
