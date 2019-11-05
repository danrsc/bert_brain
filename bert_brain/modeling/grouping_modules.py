from collections import OrderedDict

import numpy as np
import torch
import torch.nn
from .graph_part import GraphPart


__all__ = ['at_most_one_data_id', 'k_data_ids', 'GroupConcatFixedGroupSize', 'GroupPool',
           'MarkedTokenConcatFixedNumTokens']


def at_most_one_data_id(data_ids, return_first_index=False, return_last_index=False):

    if len(data_ids.size()) != 2:
        raise ValueError('data_ids must be 2D')

    maxes, _ = torch.max(data_ids, dim=1)
    repeated_maxes = torch.reshape(maxes, (-1, 1)).repeat((1, data_ids.size()[1]))
    mins, _ = torch.min(torch.where(data_ids < 0, repeated_maxes, data_ids), dim=1)

    if torch.sum(maxes != mins) > 0:
        raise ValueError('More than one data_id exists for some examples')

    if return_first_index or return_last_index:
        index_array = torch.arange(data_ids.size()[1], device=data_ids.device).view(
            (1, data_ids.size()[1])).repeat((data_ids.size()[0], 1))
        indicator_valid = data_ids >= 0
        first_index = None
        if return_first_index:
            first_index_array = torch.where(
                indicator_valid, index_array, torch.full_like(index_array, data_ids.size()[1] + 1))
            first_index, _ = torch.min(first_index_array, dim=1)
        last_index = None
        if return_last_index:
            last_index_array = torch.where(indicator_valid, index_array, torch.full_like(index_array, -1))
            last_index, _ = torch.max(last_index_array, dim=1)
        if return_first_index and return_last_index:
            return maxes, first_index, last_index
        if return_first_index:
            return maxes, first_index
        if return_last_index:
            return maxes, last_index

    return maxes


def k_data_ids(k, data_ids, return_indices=False, check_unique=False):

    if len(data_ids.size()) != 2:
        raise ValueError('data_ids must be 2D')

    indicator_valid = data_ids >= 0
    count_valid = torch.sum(indicator_valid, dim=1)
    if torch.max(count_valid) != k or torch.min(count_valid) != k:
        print(count_valid)
        raise ValueError('Incorrect number of data_ids. Expected {}'.format(k))

    data_ids = torch.masked_select(data_ids, indicator_valid)
    data_ids = torch.reshape(data_ids, (data_ids.size()[0], k))

    if check_unique:
        mins, _ = torch.min(data_ids, dim=1)
        maxes, _ = torch.max(data_ids, dim=1)
        if torch.sum(maxes != mins) > 0:
            raise ValueError('More than one data_id exists for some examples')

    if return_indices:
        index_array = torch.arange(data_ids.size()[1], device=data_ids.device).view(
            (1, data_ids.size()[1])).repeat((data_ids.size()[0], 1))
        indices = torch.masked_select(index_array, indicator_valid)
        indices = torch.reshape(indices, (indicator_valid.size()[0], k))
        return data_ids, indices

    return data_ids


class GroupBase(GraphPart):

    def __init__(self, groupby_prefixes, groupby_suffix, output_name):
        super().__init__()
        self.groupby_prefixes = groupby_prefixes
        self.groupby_suffix = groupby_suffix
        self.output_name = output_name

    @staticmethod
    def _expand_placeholders(to_expand, placeholder_name_to_fields):
        if to_expand is None:
            return None
        if np.ndim(to_expand) == 0:
            to_expand = [to_expand]
        expanded = list()
        for item in to_expand:
            if item in placeholder_name_to_fields:
                for f in placeholder_name_to_fields[item]:
                    expanded.append(f)
            else:
                expanded.append(item)
        if np.ndim(to_expand) == 0 and len(expanded) == 1:
            return expanded[0]
        return expanded

    def resolve_placeholders(self, placeholder_name_to_fields, field_shapes):
        self.groupby_prefixes = GroupBase._expand_placeholders(self.groupby_prefixes, placeholder_name_to_fields)

    def _get_group_ids(self, batch):
        groupby_prefixes = [self.groupby_prefixes] if np.ndim(self.groupby_prefixes) == 0 else self.groupby_prefixes
        all_group_ids = [batch[(groupby_prefix, self.groupby_suffix)] for groupby_prefix in groupby_prefixes]
        for idx in range(1, len(all_group_ids)):
            if not torch.equal(all_group_ids[0], all_group_ids[idx]):
                raise ValueError('Inconsistent group ids cannot be used within the same instance of {}'.format(
                    type(self)))
        return all_group_ids[0]

    def _num_channels(self, name_to_num_channels):
        raise NotImplementedError('{} does not implement _num_channels'.format(type(self)))

    def instantiate(self, name_to_num_channels):
        in_channels = self._num_channels(name_to_num_channels)
        result = OrderedDict()
        result[self.output_name] = in_channels
        result[(self.output_name, self.groupby_suffix)] = ()
        result[(self.output_name, 'example_ids')] = ()
        return result

    def _forward(self, batch, group_ids):
        raise NotImplementedError('{} does not implement forward'.format(type(self)))

    def forward(self, batch):
        x, groupby, example_ids = self._forward(batch, self._get_group_ids(batch))
        result = OrderedDict()
        result[self.output_name] = x
        result[(self.output_name, self.groupby_suffix)] = groupby
        result[(self.output_name, 'example_ids')] = example_ids
        return result


class GroupConcatFixedGroupSize(GroupBase):

    def __init__(
            self,
            num_per_group,
            groupby_prefixes,
            groupby_suffix,
            output_name,
            sequence_source_name,
            pooled_source_name=None):
        super().__init__(groupby_prefixes, groupby_suffix, output_name)
        self.num_per_group = num_per_group
        self.sequence_source_name = sequence_source_name
        self.pooled_source_name = pooled_source_name

    def _num_channels(self, name_to_num_channels):
        num_channels = name_to_num_channels[self.sequence_source_name] * self.num_per_group
        if self.pooled_source_name is not None:
            num_channels += name_to_num_channels[self.pooled_source_name]
        return num_channels

    def _forward(self, batch, groupby):

        x = batch[self.sequence_source_name]

        # first attach an example_id to the groups to ensure that we don't concat across examples in the batch

        # array of shape (batch, sequence, 1) which identifies example
        example_ids = torch.arange(
            groupby.size()[0], device=x.device).view((groupby.size()[0], 1, 1)).repeat((1, groupby.size()[1], 1))

        # indices to ensure stable sort, and to give us indices_sort
        indices = torch.arange(groupby.size()[0] * groupby.size()[1], device=x.device).view(groupby.size() + (1,))

        # -> (batch, sequence, 3): attach example_id to each group and add indices to guarantee stable sort
        groupby = torch.cat((example_ids, groupby.view(groupby.size() + (1,)), indices), dim=2)

        # -> (batch * sequence, 3)
        groupby = groupby.view((groupby.size()[0] * groupby.size()[1], groupby.size()[2]))

        # filter out the bogus groupby
        groupby = groupby[groupby[:, 1] >= 0]

        # this allows us to sort the 3 dimensions together
        groups = torch.unique(groupby, sorted=True, dim=0)

        _, counts = torch.unique_consecutive(groups[:, :2], return_counts=True, dim=0)

        # check that the input is what we expected
        if torch.min(counts) != self.num_per_group or torch.max(counts) != self.num_per_group:
            raise ValueError('Expected exactly {} per unique groupby. min count: {}, max count: {}'.format(
                self.num_per_group, torch.min(counts), torch.max(counts)))

        # get the true groups and example_ids
        example_ids = groups[:, 0]
        indices_sort = groups[:, 2]
        groups = groups[:, 1]

        # -> (batch * sequence, n, m, ..., k)
        x = x.view((x.size()[0] * x.size()[1],) + x.size()[2:])

        # sort x so that grouped items are together
        x = x[indices_sort]

        x = x.view((x.size()[0] // self.num_per_group, self.num_per_group) + x.size()[1:])
        groups = groups.view((groups.size()[0] // self.num_per_group, self.num_per_group))
        example_ids = example_ids.view((example_ids.size()[0] // self.num_per_group, self.num_per_group))

        # all of these are the same on axis=1, so take the first
        groups = groups[:, 0]
        example_ids = example_ids[:, 0]

        x = x.view(x.size()[0], x.size()[1] * x.size()[2])
        if self.pooled_source_name is not None:
            x = torch.cat([batch[self.pooled_source_name][example_ids], x], dim=1)

        return x, groups, example_ids


class GroupPool(GroupBase):

    def __init__(
            self,
            groupby_prefixes,
            groupby_suffix,
            output_name,
            sequence_source_name):
        super().__init__(groupby_prefixes, groupby_suffix, output_name)
        self.sequence_source_name = sequence_source_name

    def _num_channels(self, name_to_num_channels):
        return name_to_num_channels[self.sequence_source_name]

    def _forward(self, batch, groupby):

        x = batch[self.sequence_source_name]

        # first attach an example_id to the groups to ensure that we don't pool across examples in the batch

        # array of shape (batch, sequence, 1) which identifies example
        example_ids = torch.arange(
            groupby.size()[0], device=x.device).view((groupby.size()[0], 1, 1)).repeat((1, groupby.size()[1], 1))
        # -> (batch, sequence, 2): attach example_id to each group
        groupby = torch.cat((example_ids, groupby.view(groupby.size() + (1,))), dim=2)

        # -> (batch * sequence, 2)
        groupby = groupby.view((groupby.size()[0] * groupby.size()[1], groupby.size()[2]))

        # each group is a (example_id, group) tuple
        groups, group_indices = torch.unique(groupby, sorted=True, return_inverse=True, dim=0)

        # split the groups into the true groups and example_ids
        example_ids = groups[:, 0]
        groups = groups[:, 1]

        # -> (batch * sequence, 1, 1, ..., 1)
        group_indices = group_indices.view((x.size()[0] * x.size()[1],) + (1,) * (len(x.size()) - 2))

        # -> (batch * sequence, n, m, ..., k)
        group_indices = group_indices.repeat((1,) + x.size()[2:])

        # -> (batch * sequence, n, m, ..., k)
        x = x.view((x.size()[0] * x.size()[1],) + x.size()[2:])

        pooled = torch.zeros((groups.size()[0],) + x.size()[1:], dtype=x.dtype, device=x.device)
        pooled.scatter_add_(dim=0, index=group_indices, src=x)

        # -> (batch * sequence)
        group_indices = group_indices[:, 0]
        counts = torch.zeros(groups.size()[0], dtype=x.dtype, device=x.device)
        counts.scatter_add_(
            dim=0, index=group_indices, src=torch.ones(len(group_indices), dtype=x.dtype, device=x.device))
        counts = counts.view(counts.size() + (1,) * len(pooled.size()[1:]))
        pooled = pooled / counts

        # filter out groups < 0
        indicator_valid = groups >= 0
        pooled = pooled[indicator_valid]
        groups = groups[indicator_valid]
        example_ids = example_ids[indicator_valid]

        return pooled, groups, example_ids


class MarkedTokenConcatFixedNumTokens(GroupBase):

    def __init__(
            self,
            num_tokens,
            marker_prefixes,
            marker_suffix,
            output_name,
            sequence_source_name,
            pooled_source_name=None):
        super().__init__(marker_prefixes, marker_suffix, output_name)
        self.num_tokens = num_tokens
        self.sequence_source_name = sequence_source_name
        self.pooled_source_name = pooled_source_name

    def _num_channels(self, name_to_num_channels):
        num_channels = name_to_num_channels[self.sequence_source_name] * self.num_tokens
        if self.pooled_source_name is not None:
            num_channels += name_to_num_channels[self.pooled_source_name]
        return num_channels

    def _forward(self, batch, marker_ids):
        marker_ids, indices = k_data_ids(self.num_tokens, marker_ids, return_indices=True, check_unique=True)
        marker_ids = marker_ids[:, 0]
        if len(batch[self.sequence_source_name].size()) > indices.size():
            indices = torch.reshape(
                indices, indices.size() + (1,) * (len(batch[self.sequence_source_name].size() - len(indices.size()))))

        gathered_outputs = torch.gather(batch[self.sequence_source_name], dim=1, index=indices)
        if self.pooled_source_name:
            gathered_outputs = torch.cat(
                [torch.unsqueeze(batch[self.pooled_source_name], dim=1), gathered_outputs], dim=1)

        result = OrderedDict()
        result[self.output_name] = gathered_outputs
        result[(self.output_name, self.output_marker_name)] = marker_ids
        result[(self.output_name, 'example_ids')] = torch.arange(len(marker_ids), device=marker_ids.device)
        return result
