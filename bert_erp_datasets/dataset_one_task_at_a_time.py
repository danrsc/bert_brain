import dataclasses
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import TensorDataset

from bert_erp_common import SwitchRemember
from .input_features import FieldSpec

__all__ = ['PreparedDataDatasetOneTaskAtATime', 'BatchOneTaskRandomSampler', 'BatchOneTaskSequentialSampler']


def _pad(to_pad, sequence_length, value=0):
    if len(to_pad) < sequence_length:
        return np.pad(to_pad, (0, sequence_length - len(to_pad)), mode='constant', constant_values=value)
    return to_pad


def _pad_tokens(tokens, sequence_length, value='[PAD]'):
    if len(tokens) < sequence_length:
        return list(tokens) + [value] * (sequence_length - len(tokens))
    return list(tokens)


def _filled_values(indices, values, sequence_length, fill_with):
    indices = _pad(indices, sequence_length, value=-1)
    valid_indices = indices[indices >= 0]
    vals = np.full((sequence_length,) + values.shape[1:], fill_with)
    vals[indices >= 0] = values[valid_indices]
    return vals


def _at_most_one_value(indices, values, fill_with):
    valid_indices = indices[indices >= 0]
    if len(valid_indices) == 0:
        return np.full(values.shape[1:], fill_with)
    if len(valid_indices) > 1:
        raise ValueError('Too many non-zero indices')
    return values[valid_indices[0]]


class PreparedDataDatasetOneTaskAtATime(torch.utils.data.Dataset):

    data_set_id_field = 'data_set_id'

    @staticmethod
    def _get_examples(which, current):
        which = SwitchRemember(which)
        if which == 'train':
            return current.train if current.train is not None else []
        elif which == 'validation':
            return current.validation if current.validation is not None else []
        elif which == 'test':
            return current.test if current.test is not None else []
        raise ValueError(
            'Unknown value for which: {}. Valid choices are: ({})'.format(which.var, ', '.join(which.tests)))

    @staticmethod
    def _add_field_or_check_consistent(field_specs, to_add, corpus_field_specs):
        # get the field_spec or create a default
        if corpus_field_specs is not None and to_add in corpus_field_specs:
            field_spec = corpus_field_specs[to_add]
        else:
            field_spec = FieldSpec()

        if to_add in field_specs:
            # validate that there is no conflict between the field_spec in the current data-set
            # and previously seen field_spec
            if field_spec != field_specs[to_add]:
                raise ValueError('FieldSpec conflict on field {}: {}, {}'.format(
                    to_add, field_spec, field_specs[to_add]))
            return False

        field_specs[to_add] = field_spec
        return True

    @staticmethod
    def _is_field_allowed(filter_when_not_in_loss_keys, loss_keys, field_name, kind):
        return (
            filter_when_not_in_loss_keys is None
            or (field_name not in filter_when_not_in_loss_keys and kind not in filter_when_not_in_loss_keys)
            or field_name in loss_keys)

    def __init__(
            self,
            max_sequence_length,
            prepared_data,
            loss_keys,
            which='train',
            token_field='tokens',
            id_field='unique_id',
            data_index_field='data_ids',
            data_id_in_batch_keys=None,
            filter_when_not_in_loss_keys=None):

        self._field_specs = dict()

        self._num_examples = OrderedDict()

        self._example_tensors = OrderedDict()

        self._response_data = OrderedDict()
        self._response_data_indices = OrderedDict()
        self._response_data_kind = OrderedDict()
        self._response_data_example_counts = OrderedDict()

        self._data_id_to_tokens = dict()
        self._data_set_id_to_data_set_key = dict()
        self._field_to_data_set_key = dict()

        self._data_id_in_batch_keys = None
        if data_id_in_batch_keys is not None:
            self._data_id_in_batch_keys = set(data_id_in_batch_keys)

        self._max_sequence_length = max_sequence_length

        for data_set_id, data_key in enumerate(prepared_data):

            # add a special field to track which data-set
            self._example_tensors[data_key] = OrderedDict()
            self._example_tensors[data_key][PreparedDataDatasetOneTaskAtATime.data_set_id_field] = list()
            self._field_specs[PreparedDataDatasetOneTaskAtATime.data_set_id_field] = FieldSpec(
                fill_value=-1, tensor_dtype=torch.long, is_sequence=False)

            self._response_data_indices[data_key] = OrderedDict()

            self._data_set_id_to_data_set_key[data_set_id] = data_key

            current = prepared_data[data_key]
            examples = PreparedDataDatasetOneTaskAtATime._get_examples(which, current)

            for key in current.data:
                if key in self._response_data:
                    raise ValueError('Multiple corpora use the same key in data: {}'.format(key))
                if not PreparedDataDatasetOneTaskAtATime._is_field_allowed(
                        filter_when_not_in_loss_keys, loss_keys, key, current.data[key].kind):
                    continue

                self._field_to_data_set_key[key] = data_key

            fields_as_none = set()

            for index_example, f in enumerate(examples):
                if index_example == 0:
                    fields = [field.name for field in dataclasses.fields(f)]
                    for field in fields:

                        if current.field_specs[field].tensor_dtype == str:
                            continue

                        if not PreparedDataDatasetOneTaskAtATime._is_field_allowed(
                                filter_when_not_in_loss_keys, loss_keys, field, kind=None):
                            continue

                        # this is an optional field; we know that the field is either always None for a given dataset
                        # or it is always not None for a given dataset (but we will validate this below)
                        if getattr(f, field) is None:
                            fields_as_none.add(field)
                            continue

                        if field == data_index_field:
                            # keep a field spec for indexing into data
                            PreparedDataDatasetOneTaskAtATime._add_field_or_check_consistent(
                                self._field_specs, field, current.field_specs)
                            for response_data_key in current.data:
                                if not PreparedDataDatasetOneTaskAtATime._is_field_allowed(
                                        filter_when_not_in_loss_keys,
                                        loss_keys, response_data_key, current.data[response_data_key].kind):
                                    continue
                                if not PreparedDataDatasetOneTaskAtATime._add_field_or_check_consistent(
                                        self._field_specs, response_data_key, current.field_specs):
                                    raise ValueError('Field name conflict: {}'.format(response_data_key))
                                self._response_data_indices[data_key][response_data_key] = list()
                        else:
                            is_added = PreparedDataDatasetOneTaskAtATime._add_field_or_check_consistent(
                                self._field_specs, field, current.field_specs)
                            is_example_field = any(field in self._example_tensors[k] for k in self._example_tensors)
                            if not is_example_field:
                                if not is_added:
                                    # could happen if there is a conflict between response_data and example fields
                                    raise ValueError('Field name conflict: {}'.format(field))
                                self._example_tensors[data_key][field] = list()

                # add the current example
                example_values = dataclasses.asdict(f)
                for field in self._example_tensors[data_key]:
                    if field == PreparedDataDatasetOneTaskAtATime.data_set_id_field:
                        self._example_tensors[data_key][field].append(data_set_id)
                        continue
                    if field in example_values:
                        example_value = example_values[field]
                        if self._field_specs[field].is_sequence:
                            example_value = _pad(
                                example_value, max_sequence_length, self._field_specs[field].fill_value)
                    else:
                        raise ValueError('Inconsistent fields in dataset: {}'.format(data_key))
                    self._example_tensors[data_key][field].append(example_value)

                response_data_indices = example_values[data_index_field]
                for response_data_key in self._response_data_indices[data_key]:
                    if response_data_key in response_data_indices:
                        indices = response_data_indices[response_data_key]
                    else:
                        if self._field_specs[data_index_field].is_sequence:
                            indices = np.array([self._field_specs[data_index_field].fill_value])
                        else:  # not sure when this would happen, but we allow it
                            indices = self._field_specs[data_index_field].fill_value
                    if self._field_specs[data_index_field].is_sequence:
                        indices = _pad(indices, max_sequence_length, self._field_specs[data_index_field].fill_value)
                    if np.any(indices >= 0):
                        if response_data_key not in self._response_data_example_counts:
                            self._response_data_example_counts[response_data_key] = 1
                        else:
                            self._response_data_example_counts[response_data_key] += 1
                    self._response_data_indices[data_key][response_data_key].append(indices)

                for field in fields_as_none:
                    if field in example_values and example_values[field] is not None:
                        raise ValueError('Fields must always be set to None in a given dataset '
                                         'if they are ever None in that dataset')

                for response_data_key in current.data:
                    if not PreparedDataDatasetOneTaskAtATime._is_field_allowed(
                            filter_when_not_in_loss_keys, loss_keys, response_data_key,
                            current.data[response_data_key].kind):
                        continue
                    self._response_data[response_data_key] = current.data[response_data_key].data
                    self._response_data_kind[response_data_key] = current.data[response_data_key].kind

                # remember the tokens
                self._data_id_to_tokens[(data_set_id, example_values[id_field])] = example_values[token_field]

            self._num_examples[data_key] = len(examples)

        for data_key in self._example_tensors:
            for key in self._example_tensors[data_key]:
                self._example_tensors[data_key][key] = torch.tensor(
                    self._example_tensors[data_key][key], dtype=self._field_specs[key].tensor_dtype)

        # we don't need this anymore, and it could be confusing later if we use field specs as the canonical
        # store of what fields we have
        if data_index_field in self._field_specs:
            del self._field_specs[data_index_field]

    @property
    def fields(self):
        result = list()
        for dk in self._example_tensors:
            for k in self._example_tensors[dk]:
                result.append(k)
        for response_data_key in self._response_data:
            result.append(response_data_key)
        return tuple(result)

    @property
    def max_sequence_length(self):
        return self._max_sequence_length

    def is_response_data(self, field):
        if field not in self._field_specs:
            raise KeyError('Invalid field: {}'.format(field))
        return field in self._response_data

    def value_shape(self, field):
        if field not in self._field_specs:
            raise KeyError('Invalid field: {}'.format(field))
        for data_key in self._example_tensors:
            if field in self._example_tensors[data_key]:
                size = self._example_tensors[data_key][field].size()
                if self._field_specs[field].is_sequence:
                    return size[2:]
                return size[1:]
        size = self._response_data[field].shape  # this is a numpy array, not a torch tensor
        # response data is flat regardless of whether the field is a sequence (until we access it during batching)
        return size[1:]

    def response_data_kind(self, field):
        if field not in self._field_specs:
            raise KeyError('Unknown field: {}'.format(field))
        return self._response_data_kind[field] if field in self._response_data_kind else None

    def is_sequence(self, field):
        return self._field_specs[field].is_sequence

    def fill_value(self, field):
        return self._field_specs[field].fill_value

    def _data_keys_and_index(self, index):
        for data_key in self._example_tensors:
            for response_data_key in self._response_data_indices[data_key]:
                if index < len(self._response_data_indices[data_key][response_data_key]):
                    return data_key, response_data_key, index
                index -= len(self._example_tensors[data_key][response_data_key])
        raise IndexError('Index out of bounds: {}'.format(index))

    def __getitem__(self, item):
        data_key, response_data_key, item = self._data_keys_and_index(item)
        result = OrderedDict((k, self._example_tensors[data_key][k][item]) for k in self._example_tensors[data_key])
        # we assemble the response data JIT to reduce the memory footprint
        response_data_indices = self._response_data_indices[data_key][response_data_key][item]
        if self._data_id_in_batch_keys is not None and (
                response_data_key in self._data_id_in_batch_keys
                or self._response_data_kind[response_data_key] in self._data_id_in_batch_keys):
            result[(response_data_key, 'data_ids')] = torch.tensor(
                _pad(response_data_indices, self.max_sequence_length, value=-1), dtype=torch.long)
        else:
            if self.is_sequence(response_data_key):
                response_data = _filled_values(
                    response_data_indices,
                    self._response_data[response_data_key],
                    self.max_sequence_length,
                    self._field_specs[response_data_key].fill_value)
            else:
                response_data = _at_most_one_value(
                    response_data_indices,
                    self._response_data[response_data_key],
                    self._field_specs[response_data_key].fill_value)
            result[response_data_key] = torch.tensor(
                response_data, dtype=self._field_specs[response_data_key].tensor_dtype)

        return result

    def get_data_for_data_ids(self, field, data_ids):
        if field not in self._response_data:
            raise KeyError('Field is not a response data field: {}'.format(field))
        data_ids = np.asarray(data_ids)
        return torch.tensor(self._response_data[field][data_ids], dtype=self._field_specs[field].tensor_dtype)

    def __len__(self):
        return sum(self.task_lengths())

    def task_lengths(self):
        lengths = list()
        for data_key in self._example_tensors:
            for response_data_key in self._response_data_indices[data_key]:
                lengths.append(len(self._response_data_indices[data_key][response_data_key]))
        return lengths

    def data_set_key_for_id(self, data_set_id):
        if isinstance(data_set_id, torch.Tensor):
            data_set_id = data_set_id.cpu().item()
        elif isinstance(data_set_id, np.ndarray):
            data_set_id = data_set_id.item()
        return self._data_set_id_to_data_set_key[data_set_id]

    def data_set_key_for_field(self, field):
        if field in self._field_to_data_set_key:
            return self._field_to_data_set_key[field]
        return None

    def get_tokens(self, data_set_id, item_id):
        if isinstance(data_set_id, torch.Tensor):
            data_set_id = data_set_id.cpu().item()
        elif isinstance(data_set_id, np.ndarray):
            data_set_id = data_set_id.item()
        if isinstance(item_id, torch.Tensor):
            item_id = item_id.cpu().item()
        elif isinstance(item_id, np.ndarray):
            item_id = item_id.item()
        key = (data_set_id, item_id)
        if key not in self._data_id_to_tokens:
            if data_set_id not in self._data_set_id_to_data_set_key:
                raise ValueError('Invalid data_set_id: {}'.format(data_set_id))
            data_set_key = self._data_set_id_to_data_set_key[data_set_id]
            raise KeyError('Item does not exist in dataset: {}, {}'.format(data_set_key, item_id))
        return self._data_id_to_tokens[key]

    def num_examples_for_field(self, field):
        if field not in self._field_specs:
            raise KeyError('Unknown field: {}'.format(field))
        if field in self._response_data_example_counts:
            return self._response_data_example_counts[field]
        return len(self)


class BatchOneTaskRandomSampler(torch.utils.data.Sampler):

    def __init__(self, data_source: PreparedDataDatasetOneTaskAtATime, batch_size, drop_last):
        super().__init__(data_source)
        self.task_lengths = np.array(data_source.task_lengths())
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        offset = 0
        batches = list()
        for task_length in self.task_lengths:
            task_sample = np.random.permutation(task_length) + offset
            task_sample = np.array_split(task_sample, int(np.ceil(len(task_sample) / self.batch_size)))
            if len(task_sample[-1]) < self.batch_size and self.drop_last:
                task_sample = task_sample[:-1]
            batches.extend(task_sample)
            offset += task_length
        batches = np.random.permutation(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        result = 0
        for task_length in self.task_lengths:
            if self.drop_last:
                result += task_length // self.batch_size
            else:
                result += int(np.ceil(task_length / self.batch_size))
        return result


class BatchOneTaskSequentialSampler(torch.utils.data.Sampler):

    def __init__(self, data_source: PreparedDataDatasetOneTaskAtATime, batch_size, drop_last):
        super().__init__(data_source)
        self.task_lengths = np.array(data_source.task_lengths())
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        offset = 0
        for task_length in self.task_lengths:
            task_sample = np.arange(task_length) + offset
            task_sample = np.array_split(task_sample, int(np.ceil(len(task_sample) / self.batch_size)))
            if len(task_sample[-1]) < self.batch_size and self.drop_last:
                task_sample = task_sample[:-1]
            for batch in task_sample:
                yield batch
            offset += task_length

    def __len__(self):
        result = 0
        for task_length in self.task_lengths:
            if self.drop_last:
                result += task_length // self.batch_size
            else:
                result += int(np.ceil(task_length / self.batch_size))
        return result
