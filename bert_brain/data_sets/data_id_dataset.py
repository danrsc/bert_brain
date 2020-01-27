import os
import pickle
import dataclasses
from collections import OrderedDict
from typing import Mapping, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset
# noinspection PyUnresolvedReferences
from torch.utils.data.dataloader import default_collate
# noinspection PyProtectedMember
from torch._six import container_abcs

from ..common import SwitchRemember
from .input_features import FieldSpec, InputFeatures

__all__ = [
    'collate_fn',
    'DataIdDataset']


def _pad(to_pad, sequence_length, value=0):
    if len(to_pad) < sequence_length:
        return np.pad(to_pad, (0, sequence_length - len(to_pad)), mode='constant', constant_values=value)
    return to_pad


def _pad_tokens(tokens, sequence_length, value='[PAD]'):
    if len(tokens) < sequence_length:
        return list(tokens) + [value] * (sequence_length - len(tokens))
    return list(tokens)


@dataclasses.dataclass(frozen=True)
class DataIdDatasetInit:
    data_set_key: str
    max_sequence_length: int
    field_specs: Mapping[str, FieldSpec]
    value_shapes: Mapping[str, Tuple[int, ...]]
    unique_id_to_multipart_id: Mapping[int, int]
    data_kinds: Mapping[str, str]
    has_word_ids: Mapping[str, bool]
    data_key_to_unique_ids: Mapping[str, Mapping[str, np.ndarray]]


def _packed_response_collate(batch):
    elem = batch[0]
    assert(isinstance(elem, torch.Tensor))
    return torch.cat(batch, 0)


def collate_fn(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, container_abcs.Mapping) and elem_type.__module__ != 'numpy':
        result = elem_type()
        for k in batch[0]:
            if isinstance(k, str) and k.startswith('__packed_response_data__.'):
                result[k] = _packed_response_collate([d[k] for d in batch])
            else:
                result[k] = collate_fn([d[k] for d in batch])
        return result
    else:
        return default_collate(batch)


class DataIdDataset(torch.utils.data.Dataset):

    example_file_format = '{data_set_key}_{which}_{unique_id}.pkl'
    dataset_init_file = 'init.pkl'
    data_file_format = '{data_set_key}_{response_data_key}_data_{data_id}.pkl'
    word_ids_file_format = '{data_set_key}_{response_data_key}_word_ids_{data_id}.pkl'
    metadata_file_format = '{data_set_key}_metadata_{metadata_key}_{data_id}.pkl'

    unique_id_field = 'unique_id'
    data_id_field = 'data_ids'
    word_ids_field = 'word_ids'
    multipart_id_field = 'multipart_id'
    response_id_field = 'response_id'
    packed_response_data_field_fmt = '__packed_response_data__.{response_data_key}.{what}'

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
    def _is_field_allowed(filter_when_not_in_loss_keys, loss_keys, field_name, kind):
        return (
            filter_when_not_in_loss_keys is None
            or (field_name not in filter_when_not_in_loss_keys and kind not in filter_when_not_in_loss_keys)
            or field_name in loss_keys
            or kind in loss_keys)

    def __init__(
            self,
            path,
            which,
            pad_sequence_length,
            loss_keys,
            data_id_in_batch_keys=None,
            filter_when_not_in_loss_keys=None,
            index_responses_separately=True,
            ignore_multipart_ids=False):

        self._path = path
        self._which = which
        self._pad_sequence_length = pad_sequence_length
        self._index_responses_separately = index_responses_separately

        init_data = DataIdDataset.get_init_metadata(path)
        self._data_set_key = init_data.data_set_key
        self._max_sequence_length = init_data.max_sequence_length
        self._field_specs = OrderedDict()
        self._data_id_field_spec = None
        self._value_shapes = dict()
        for field in init_data.field_specs:
            if field == DataIdDataset.data_id_field:
                self._data_id_field_spec = init_data.field_specs[field]
                continue
            if init_data.field_specs[field].tensor_dtype == str:
                continue
            kind = init_data.data_kinds[field] if field in init_data.data_kinds else None
            if not DataIdDataset._is_field_allowed(filter_when_not_in_loss_keys, loss_keys, field, kind=kind):
                continue
            self._field_specs[field] = init_data.field_specs[field]
            self._value_shapes[field] = init_data.value_shapes[field]
        unique_id_to_multipart_id = init_data.unique_id_to_multipart_id if not ignore_multipart_ids else None
        self._response_data_kind = init_data.data_kinds
        self._response_data_has_word_ids = init_data.has_word_ids
        self._response_data_key_to_unique_ids = init_data.data_key_to_unique_ids[which]
        all_unique_ids = set()
        for response_data_key in self._response_data_key_to_unique_ids:
            all_unique_ids.update(self._response_data_key_to_unique_ids[response_data_key])
        self._all_unique_ids = np.array(sorted(all_unique_ids))

        self._multipart_indices = None
        self._all_multipart_indices = None
        if unique_id_to_multipart_id is not None:
            self._multipart_indices = dict()
            for response_key in self._response_data_key_to_unique_ids:
                multipart_indices = dict()
                for index_unique_id, unique_id in enumerate(self._response_data_key_to_unique_ids[response_key]):
                    multipart_id = unique_id_to_multipart_id[unique_id]
                    if multipart_id not in multipart_indices:
                        multipart_indices[multipart_id] = set()
                    multipart_indices[multipart_id].add(index_unique_id)
                multipart_indices = [sorted(multipart_indices[m]) for m in multipart_indices]
                self._multipart_indices[response_key] = tuple(
                    sorted(multipart_indices, key=lambda indices_: indices_[0]))
            all_multipart_indices = dict()
            for index_unique_id, unique_id in enumerate(self._all_unique_ids):
                multipart_id = unique_id_to_multipart_id[unique_id]
                if multipart_id not in all_multipart_indices:
                    all_multipart_indices[multipart_id] = set()
                all_multipart_indices[multipart_id].add(index_unique_id)
            all_multipart_indices = [sorted(all_multipart_indices[m]) for m in all_multipart_indices]
            self._all_multipart_indices = tuple(sorted(all_multipart_indices, key=lambda indices_: indices_[0]))

        self._data_id_in_batch_keys = set(data_id_in_batch_keys) if data_id_in_batch_keys is not None else set()

    @staticmethod
    def get_init_metadata(path: str) -> DataIdDatasetInit:
        init_file_path = os.path.join(path, DataIdDataset.dataset_init_file)
        with open(init_file_path, 'rb') as init_file:
            return pickle.load(init_file)

    @staticmethod
    def make_dataset_files(path, data_set_key, prepared_data, metadata):
        max_sequence_length = 0
        fields_as_none = set()
        self_field_specs = None
        self_unique_ids = dict()
        self_unique_id_to_multipart_id = None
        self_data_kinds = OrderedDict()
        self_value_shapes = dict()
        self_has_word_ids = dict()

        if not os.path.exists(path):
            os.makedirs(path)

        for response_data_key in prepared_data.data:
            self_data_kinds[response_data_key] = prepared_data.data[response_data_key].kind
            for data_id, item in enumerate(prepared_data.data[response_data_key].data):
                data_file_path = os.path.join(
                    path, DataIdDataset.data_file_format.format(
                        data_set_key=data_set_key, response_data_key=response_data_key, data_id=data_id))
                with open(data_file_path, 'wb') as data_file:
                    pickle.dump(item, data_file, protocol=pickle.HIGHEST_PROTOCOL)

            if prepared_data.data[response_data_key].word_ids is not None:
                self_has_word_ids[response_data_key] = True
                for data_id, item in enumerate(prepared_data.data[response_data_key].word_ids):
                    word_ids_file_path = os.path.join(
                        path, DataIdDataset.word_ids_file_format.format(
                            data_set_key=data_set_key, response_data_key=response_data_key, data_id=data_id))
                    with open(word_ids_file_path, 'wb') as word_ids_file:
                        pickle.dump(item, word_ids_file, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                self_has_word_ids[response_data_key] = False

        if metadata is not None:
            for metadata_key in metadata:
                for data_id, item in enumerate(metadata[metadata_key]):
                    metadata_file_path = os.path.join(path, DataIdDataset.metadata_file_format.format(
                        data_set_key=data_set_key, metadata_key=metadata_key, data_id=data_id))
                    with open(metadata_file_path, 'wb') as metadata_file:
                        pickle.dump(item, metadata_file, protocol=pickle.HIGHEST_PROTOCOL)

        for which in ('train', 'test', 'validation'):
            examples = DataIdDataset._get_examples(which, prepared_data)
            self_unique_ids[which] = dict()
            for index_example, example in enumerate(examples):
                example_values = dataclasses.asdict(example)
                if self_field_specs is None:
                    self_field_specs = OrderedDict()
                    for field in example_values:
                        if field in self_data_kinds:
                            raise ValueError('response_data_field conflicts with example field: {}'.format(field))
                        if example_values[field] is None:
                            fields_as_none.add(field)
                        else:
                            self_field_specs[field] = prepared_data.field_specs[field] \
                                if prepared_data.field_specs is not None and field in prepared_data.field_specs \
                                else FieldSpec()
                            if self_field_specs[field].is_sequence:
                                self_value_shapes[field] = np.shape(example_values[field])[1:]
                            else:
                                self_value_shapes[field] = np.shape(example_values[field])
                            if field == DataIdDataset.multipart_id_field:
                                self_unique_id_to_multipart_id = dict()
                    for field in self_data_kinds:
                        self_field_specs[field] = prepared_data.field_specs[field] \
                            if prepared_data.field_specs is not None and field in prepared_data.field_specs \
                            else FieldSpec()
                        # response_data is flat at this point regardless of whether the field is a sequence
                        self_value_shapes[field] = prepared_data.data[field].data.shape[1:]

                for field in example_values:
                    # make sure optional fields are either always None or never None
                    if (example_values[field] is None) != (field in fields_as_none):
                        raise ValueError(
                            'Fields must be always None or never None. Violated by: {}'.format(field))
                    if example_values[field] is not None:
                        have_shape = self_value_shapes[field]
                        current_shape = np.shape(example_values[field])
                        if self_field_specs[field].is_sequence:
                            current_shape = current_shape[1:]
                        if not np.array_equal(have_shape, current_shape):
                            raise ValueError('Inconsistent shapes ({}, {}) for field: {}'.format(
                                have_shape, current_shape, field))

                max_sequence_length = max(max_sequence_length, len(example.tokens))

                file_path = os.path.join(
                    path, DataIdDataset.example_file_format.format(
                        data_set_key=data_set_key, which=which, unique_id=example.unique_id))

                with open(file_path, 'wb') as example_file:
                    pickle.dump(example, example_file, protocol=pickle.HIGHEST_PROTOCOL)

                for response_data_key in example.data_ids:
                    if np.any(example.data_ids[response_data_key] >= 0):
                        if response_data_key not in self_unique_ids[which]:
                            self_unique_ids[which][response_data_key] = list()
                        self_unique_ids[which][response_data_key].append(example.unique_id)
                        if self_unique_id_to_multipart_id is not None:
                            self_unique_id_to_multipart_id[example.unique_id] = example.multipart_id

        init_data = DataIdDatasetInit(
            data_set_key=data_set_key,
            max_sequence_length=max_sequence_length,
            field_specs=self_field_specs,
            value_shapes=self_value_shapes,
            unique_id_to_multipart_id=self_unique_id_to_multipart_id,
            data_kinds=self_data_kinds,
            has_word_ids=self_has_word_ids,
            data_key_to_unique_ids=self_unique_ids)

        init_file_path = os.path.join(path, DataIdDataset.dataset_init_file)
        with open(init_file_path, 'wb') as init_file:
            pickle.dump(init_data, init_file, protocol=pickle.HIGHEST_PROTOCOL)

    @property
    def data_set_key(self):
        return self._data_set_key

    @property
    def max_sequence_length(self):
        return self._max_sequence_length

    @property
    def fields(self):
        return tuple(self._field_specs)

    @property
    def response_fields(self):
        return tuple(self._response_data_kind)

    def is_response_data(self, field, allow_invalid_field=False):
        if field not in self._field_specs:
            if allow_invalid_field:
                return False
            raise KeyError('Invalid field: {}'.format(field))
        return field in self._response_data_kind

    def response_data_kind(self, field):
        if field not in self._field_specs:
            raise KeyError('Unknown field: {}'.format(field))
        return self._response_data_kind[field] if field in self._response_data_kind else None

    def is_sequence(self, field):
        return self._field_specs[field].is_sequence

    def fill_value(self, field):
        return self._field_specs[field].fill_value

    def value_shape(self, field):
        return self._value_shapes[field]

    def just_in_time_targets(self, batch, predictions):
        # fetch the data that was too expensive to put in batch as padded
        for k in predictions:
            if (isinstance(k, tuple)
                    and len(k) == 2
                    and k[0] not in batch
                    and k[0] in predictions
                    and k[1] == DataIdDataset.data_id_field
                    and self.is_response_data(k[0], allow_invalid_field=True)):

                packed_data_id = DataIdDataset.packed_response_data_field_fmt.format(
                    response_data_key=k[0], what='data_ids')
                packed_data = DataIdDataset.packed_response_data_field_fmt.format(
                    response_data_key=k[0], what='data')
                packed_word_ids = DataIdDataset.packed_response_data_field_fmt.format(
                    response_data_key=k[0], what='word_ids')

                if packed_data_id not in batch:
                    raise KeyError('Packed response data does not exist for field: {}'.format(k[0]))
                data_id_to_index = dict((d.item(), i) for i, d in enumerate(batch[packed_data_id]))
                indices = np.array(list(data_id_to_index[d.item()] for d in predictions[k]))
                group_data = batch[packed_data][indices]
                word_ids = None
                if packed_word_ids in batch:
                    word_ids = batch[packed_word_ids][indices]

                batch[k[0]] = group_data.to(predictions[k[0]].device)
                if word_ids is not None:
                    batch[(k[0], DataIdDataset.word_ids_field)] = torch.as_tensor(
                        word_ids, device=predictions[k[0]].device)

    def _response_key_and_index(self, index):
        for response_id, response_key in enumerate(self._response_data_kind):
            if index < len(self._response_data_key_to_unique_ids[response_key]):
                return response_id, response_key, index
            index -= len(self._response_data_key_to_unique_ids[response_key])
        raise IndexError('Index out of bounds: {}'.format(index))

    def __getitem__(self, item):
        if self._index_responses_separately:
            response_id, response_data_key_, item = self._response_key_and_index(item)
            response_data_keys = [response_data_key_]
            unique_id = self._response_data_key_to_unique_ids[response_data_key_][item]
        else:
            response_id = None
            response_data_keys = self._response_data_kind
            unique_id = self._all_unique_ids[item]

        example_values = dataclasses.asdict(self.get_input_features(unique_id))
        result = OrderedDict()

        if response_id is not None:
            result[DataIdDataset.response_id_field] = response_id

        for field in example_values:
            if field not in self._field_specs:
                continue
            if self._field_specs[field].is_sequence:
                result[field] = torch.tensor(
                    _pad(example_values[field], self._pad_sequence_length, self._field_specs[field].fill_value),
                    dtype=self._field_specs[field].tensor_dtype)
            else:
                result[field] = torch.tensor(example_values[field], dtype=self._field_specs[field].tensor_dtype)

        response_data_indices = example_values[DataIdDataset.data_id_field]
        for response_data_key in response_data_keys:
            if response_data_key in response_data_indices:
                indices = response_data_indices[response_data_key]
            else:
                if self.is_sequence(response_data_key):
                    indices = np.array([self._data_id_field_spec.fill_value])
                else:  # not sure when this would happen, but we allow it
                    indices = self._data_id_field_spec.fill_value
            if self.is_sequence(response_data_key):
                indices = _pad(indices, self._pad_sequence_length, self._data_id_field_spec.fill_value)

            data_items = list()
            word_ids = list() if self._response_data_has_word_ids[response_data_key] else None
            for data_id in indices:
                data_item = None
                item_word_ids = None
                if data_id >= 0:
                    data_item = self.get_data_item(response_data_key, data_id)
                    if word_ids is not None:
                        item_word_ids = self.get_word_ids(response_data_key, data_id)
                data_items.append(data_item)
                if word_ids is not None:
                    word_ids.append(item_word_ids)

            if self._data_id_in_batch_keys is not None and (
                    response_data_key in self._data_id_in_batch_keys
                    or self._response_data_kind[response_data_key] in self._data_id_in_batch_keys):
                data_items = list(d for d, i in zip(data_items, indices) if i >= 0)
                if word_ids is not None:
                    word_ids = list(w for w, i in zip(word_ids, indices) if i >= 0)
                # put the data into the result packed together, with pointers on the words
                packed_data_id = DataIdDataset.packed_response_data_field_fmt.format(
                    response_data_key=response_data_key, what='data_ids')
                packed_data = DataIdDataset.packed_response_data_field_fmt.format(
                    response_data_key=response_data_key, what='data')
                packed_word_ids = DataIdDataset.packed_response_data_field_fmt.format(
                    response_data_key=response_data_key, what='word_ids')
                result[packed_data_id] = torch.tensor(
                    list(i for i in indices if i >= 0), dtype=self._data_id_field_spec.tensor_dtype)
                result[packed_data] = torch.tensor(
                    data_items, dtype=self._field_specs[response_data_key].tensor_dtype)
                if word_ids is not None:
                    result[packed_word_ids] = torch.tensor(
                        word_ids, dtype=self._data_id_field_spec.tensor_dtype)
                # put pointers
                result[(response_data_key, DataIdDataset.data_id_field)] = torch.tensor(
                    indices, dtype=self._data_id_field_spec.tensor_dtype)
            else:
                if self.is_sequence(response_data_key):
                    data = np.full(
                        (len(indices),) + self._value_shapes[response_data_key],
                        self._field_specs[response_data_key].fill_value)
                    word_ids_ = None if word_ids is None else np.full(len(indices), self._data_id_field_spec.fill_value)
                    for index_sequence, data_id in enumerate(indices):
                        if data_id >= 0:
                            data[index_sequence] = data_items[index_sequence]
                            if word_ids_ is not None:
                                word_ids_[index_sequence] = word_ids[index_sequence]
                    result[response_data_key] = torch.tensor(
                        data, dtype=self._field_specs[response_data_key].tensor_dtype)
                    if word_ids_ is not None:
                        result[(response_data_key, DataIdDataset.word_ids_field)] = torch.tensor(
                            word_ids_, dtype=self._data_id_field_spec.tensor_dtype)
                else:
                    index_valid = None
                    for index_sequence, data_id in enumerate(indices):
                        if data_id >= 0:
                            if index_valid is not None:
                                raise ValueError('Too many valid indices for a non-sequence field: {}'.format(
                                    response_data_key))
                            index_valid = index_sequence
                    if index_valid is None:
                        result[response_data_key] = torch.tensor(
                            np.full(
                                self._value_shapes[response_data_key],
                                self._field_specs[response_data_key].fill_value),
                            dtype=self._field_specs[response_data_key].tensor_dtype)
                        if word_ids is not None:
                            result[(response_data_key, DataIdDataset.word_ids_field)] = torch.tensor(
                                self._data_id_field_spec.fill_value, dtype=self._data_id_field_spec.tensor_dtype)
                    else:
                        result[response_data_key] = torch.tensor(
                            data_items[index_valid], dtype=self._field_specs[response_data_key].tensor_dtype)
                        if word_ids is not None:
                            result[(response_data_key, DataIdDataset.word_ids_field)] = torch.tensor(
                                word_ids[index_valid], dtype=self._data_id_field_spec.tensor_dtype)

        return result

    def length(self, task_filter=None):
        task_indices = self.task_indices()
        if task_filter is not None:
            if not self._index_responses_separately:
                raise ValueError('Cannot use task_filter when index_responses_separately is False')
            return sum(sum(len(task_item_list) for task_item_list in task_indices[task])
                       for task in task_indices if task in task_filter or self.response_data_kind(task) in task_filter)
        elif self._index_responses_separately:
            return sum(sum(len(task_item_list) for task_item_list in task_indices[task]) for task in task_indices)
        else:
            return sum(len(task_item_list) for task_item_list in task_indices)

    def __len__(self):
        return self.length()

    def task_indices(self):
        if self._index_responses_separately:
            indices = OrderedDict()
            offset = 0
            for response_key in self._response_data_kind:
                num_examples = len(self._response_data_key_to_unique_ids[response_key]) \
                    if response_key in self._response_data_key_to_unique_ids else 0
                if self._multipart_indices is not None:
                    task_indices = [(np.array(i) + offset) for i in self._multipart_indices[response_key]]
                    count = sum(len(i) for i in task_indices)
                    # if this doesn't hold, we need to change the logic in _response_key_and_index
                    assert(count == num_examples)
                else:
                    if num_examples == 0:
                        task_indices = []
                    else:
                        task_indices = np.split(np.arange(num_examples) + offset, num_examples)
                indices[response_key] = task_indices
                offset += num_examples
            return indices
        else:
            num_examples = len(self._all_unique_ids)
            if self._all_multipart_indices is not None:
                task_indices = [np.array(i) for i in self._all_multipart_indices]
                count = sum(len(i) for i in task_indices)
                # if this doesn't hold, we need to change the logic in __get_item__
                assert (count == num_examples)
            else:
                if num_examples == 0:
                    task_indices = []
                else:
                    task_indices = np.split(np.arange(num_examples), num_examples)
            return task_indices

    def get_input_features(self, unique_id) -> InputFeatures:
        file_path = os.path.join(
            self._path, DataIdDataset.example_file_format.format(
                data_set_key=self.data_set_key, which=self._which, unique_id=unique_id))

        with open(file_path, 'rb') as example_file:
            return pickle.load(example_file)

    def get_tokens(self, unique_id):
        input_features = self.get_input_features(unique_id)
        return input_features.tokens

    def get_data_item(self, response_data_key, data_id):
        data_file_path = os.path.join(
            self._path, DataIdDataset.data_file_format.format(
                data_set_key=self._data_set_key, response_data_key=response_data_key, data_id=data_id))
        if not os.path.exists(data_file_path):
            raise KeyError('No data exist for {}, {}'.format(response_data_key, data_id))
        with open(data_file_path, 'rb') as data_file:
            return pickle.load(data_file)

    def get_word_ids(self, response_data_key, data_id):
        if not self._response_data_has_word_ids[response_data_key]:
            raise ValueError('No word ids exist for {}'.format(response_data_key))
        word_ids_file_path = os.path.join(
            self._path, DataIdDataset.word_ids_file_format.format(
                data_set_key=self._data_set_key, response_data_key=response_data_key, data_id=data_id))
        with open(word_ids_file_path, 'rb') as word_ids_file:
            return pickle.load(word_ids_file)

    def num_examples_for_field(self, field):
        if field not in self._field_specs:
            raise KeyError('Unknown field: {}'.format(field))
        if field in self._response_data_key_to_unique_ids:
            return len(self._response_data_key_to_unique_ids[field])
        return len(self._all_unique_ids)
