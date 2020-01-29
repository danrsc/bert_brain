import dataclasses
from collections import OrderedDict
from typing import Optional, Iterable, Container, Mapping, Any

import numpy as np
import torch
import torch.utils.data

from .data_id_dataset import DataIdDataset
from .input_features import InputFeatures

__all__ = [
    'DataIdMultiDataset',
    'SamplerFactory',
    'BatchOneTaskSamplerFactory',
    'BatchOneTaskSampler',
    'BatchOneTaskTemperatureProportionalSamplerFactory',
    'BatchOneTaskTemperatureProportionalSampler',
    'BatchOneTaskProportionalSamplerFactory',
    'BatchOneTaskSequentialSamplerFactory',
    'BatchOneTaskSequentialSampler',
    'BatchOneTaskRandomSamplerFactory',
    'BatchOneTaskRandomSampler',
    'RandomSamplerFactory']


class DataIdMultiDataset(torch.utils.data.Dataset):

    data_set_id_field = 'data_set_id'

    @staticmethod
    def _add_field_or_check_consistent(field_specs, to_add, corpus_field_specs):
        field_spec = corpus_field_specs[to_add]
        if to_add in field_specs:
            # validate that there is no conflict between the field_spec in the current data-set
            # and previously seen field_spec
            if field_spec != field_specs[to_add]:
                raise ValueError('FieldSpec conflict on field {}: {}, {}'.format(
                    to_add, field_spec, field_specs[to_add]))
            return False

        field_specs[to_add] = field_spec
        return True

    def __init__(
            self,
            which: str,
            paths: Iterable[str],
            loss_keys: Container[str],
            data_id_in_batch_keys: Optional[Iterable[str]] = None,
            filter_when_not_in_loss_keys: Optional[Iterable[str]] = None,
            field_spec_replacers: Optional[Mapping[str, Mapping[str, Any]]] = None,
            index_responses_separately: bool = True,
            ignore_multipart_ids: bool = False):

        self._max_sequence_length = max(DataIdDataset.get_init_metadata(path).max_sequence_length for path in paths)
        self._data_sets = OrderedDict()
        self._data_set_lengths = OrderedDict()
        self._data_set_id_to_data_set_key = dict()
        self._field_to_data_set_key = dict()
        self._field_specs = OrderedDict()

        for data_set_id, path in enumerate(paths):
            data_set = DataIdDataset(
                path,
                which,
                self.max_sequence_length,
                loss_keys,
                data_id_in_batch_keys,
                filter_when_not_in_loss_keys,
                field_spec_replacers,
                index_responses_separately,
                ignore_multipart_ids=ignore_multipart_ids)

            for field in data_set.response_fields:
                if field in self._field_to_data_set_key:
                    raise ValueError('Multiple corpora ({}, {}) use the same response field: {}'.format(
                        self._field_to_data_set_key[field], data_set.data_set_key, field))
                self._field_to_data_set_key[field] = data_set.data_set_key

            self._data_sets[data_set.data_set_key] = data_set
            # cache this since we use it over and over and it may take some compute
            self._data_set_lengths[data_set.data_set_key] = len(data_set)
            self._data_set_id_to_data_set_key[data_set_id] = data_set.data_set_key

            # noinspection PyProtectedMember
            for field in data_set._field_specs:
                # noinspection PyProtectedMember
                DataIdMultiDataset._add_field_or_check_consistent(
                    self._field_specs, field, data_set._field_specs)

    @property
    def max_sequence_length(self):
        return self._max_sequence_length

    @property
    def fields(self):
        return tuple(self._field_specs)

    @property
    def response_fields(self):
        r = list()
        for data_set_key in self._data_sets:
            r.extend(self._data_sets[data_set_key].response_fields)
        return tuple(r)

    def is_response_data(self, field, allow_invalid_field=False):
        if field not in self._field_specs:
            if allow_invalid_field:
                return False
            raise KeyError('Invalid field: {}'.format(field))
        try:
            if self.response_data_kind(field) is not None:
                return True
        except KeyError:
            return False

    def is_just_in_time_field(self, field, allow_invalid_field=False):
        if field not in self._field_specs:
            if allow_invalid_field:
                return False
            raise KeyError('Invalid field: {}'.format(field))
        for data_set_key in self._data_sets:
            if self._data_sets[data_set_key].is_just_in_time_field(field, allow_invalid_field=True):
                return True
        return False

    def response_data_kind(self, field):
        if field not in self._field_specs:
            raise KeyError('Unknown field: {}'.format(field))
        for data_set_key in self._data_sets:
            try:
                kind = self._data_sets[data_set_key].response_data_kind(field)
                if kind is not None:
                    return kind
            except KeyError:
                continue
        return None

    def is_sequence(self, field):
        return self._field_specs[field].is_sequence

    def fill_value(self, field):
        return self._field_specs[field].fill_value

    def value_shape(self, field):
        if field not in self._field_specs:
            raise KeyError('Unknown field: {}'.format(field))
        for data_set_key in self._data_sets:
            try:
                return self._data_sets[data_set_key].value_shape(field)
            except KeyError:
                continue

    def _data_set_key_and_index(self, index):
        response_id_offset = 0
        for data_set_id, data_set_key in enumerate(self._data_sets):
            if index < self._data_set_lengths[data_set_key]:
                return response_id_offset, data_set_id, data_set_key, index
            index -= self._data_set_lengths[data_set_key]
            response_id_offset += len(self._data_sets[data_set_key].response_fields)
        raise IndexError('Index out of bounds: {}'.format(index))

    def __getitem__(self, item):
        response_id_offset, data_set_id, data_set_key, item = self._data_set_key_and_index(item)
        result = self._data_sets[data_set_key][item]
        result[DataIdMultiDataset.data_set_id_field] = data_set_id
        if DataIdDataset.response_id_field in result:
            result[DataIdDataset.response_id_field] = result[DataIdDataset.response_id_field] + response_id_offset
        return result

    def just_in_time_targets(self, batch, predictions):
        for data_set_key in self._data_sets:
            self._data_sets[data_set_key].just_in_time_targets(batch, predictions)

    def length(self, task_filter=None):
        task_indices = self.task_indices()
        if task_filter is not None:
            if not isinstance(task_indices, dict):
                raise ValueError('Cannot use task_filter when index_responses_separately is False')
            return sum(sum(len(task_item_list) for task_item_list in task_indices[task])
                       for task in task_indices if task in task_filter or self.response_data_kind(task) in task_filter)
        elif isinstance(task_indices, dict):
            return sum(sum(len(task_item_list) for task_item_list in task_indices[task]) for task in task_indices)
        else:
            return sum(len(task_item_list) for task_item_list in task_indices)

    def __len__(self):
        return self.length()

    def task_indices(self):
        indices = None
        offset = 0
        for data_set_key in self._data_sets:
            t = self._data_sets[data_set_key].task_indices()
            if indices is None:
                indices = t
            else:
                if isinstance(t, dict) != isinstance(indices, dict):
                    # should never happen
                    raise RuntimeError('Inconsistent task_indices across data_sets')
                if isinstance(indices, dict):
                    for k in t:
                        assert(k not in indices)
                        indices[k] = list()
                        for item in t[k]:
                            indices[k].append(item + offset)
                else:
                    for item in t:
                        indices.append(item + offset)
            offset += self._data_set_lengths[data_set_key]
        return indices

    def num_examples_for_field(self, field):
        data_set = self.data_set_key_for_field(field)
        if data_set is None:
            return len(self)
        return self._data_sets[data_set].num_examples_for_field(field)

    def data_set_key_for_field(self, field) -> Optional[str]:
        if field in self._field_to_data_set_key:
            return self._field_to_data_set_key[field]
        return None

    def data_set_key_for_id(self, data_set_id) -> str:
        if isinstance(data_set_id, torch.Tensor):
            data_set_id = data_set_id.cpu().item()
        elif isinstance(data_set_id, np.ndarray):
            data_set_id = data_set_id.item()
        return self._data_set_id_to_data_set_key[data_set_id]

    def get_input_features(self, data_set_id, unique_id) -> InputFeatures:
        return self._data_sets[self.data_set_key_for_id(data_set_id)].get_input_features(unique_id)

    def get_tokens(self, data_set_id, unique_id):
        return self.get_input_features(data_set_id, unique_id).tokens


class SamplerFactory:

    @classmethod
    def is_batch_sampler(cls):
        raise NotImplementedError('{} does not implement is_batch_sampler'.format(cls))

    @classmethod
    def is_one_task_at_a_time_sampler(cls):
        raise NotImplementedError('{} does not implement is_one_task_at_a_time_sampler'.format(cls))

    def make_sampler(
            self,
            data_source: DataIdMultiDataset,
            batch_size: int,
            task_filter: Optional[Container[str]] = None):
        raise NotImplementedError('{} does not implement make_sampler'.format(type(self)))


@dataclasses.dataclass(frozen=True)
class BatchOneTaskSamplerFactory(SamplerFactory):
    batches_per_epoch: int

    @classmethod
    def is_batch_sampler(cls):
        return True

    @classmethod
    def is_one_task_at_a_time_sampler(cls):
        return True

    def make_sampler(
            self,
            data_source: DataIdMultiDataset,
            batch_size: int,
            task_filter: Optional[Container[str]] = None):
        return BatchOneTaskSampler(data_source, batch_size, self.batches_per_epoch, task_filter)


@dataclasses.dataclass(frozen=True)
class BatchOneTaskProportionalSamplerFactory(BatchOneTaskSamplerFactory):
    max_contribution: Optional[int] = None

    def make_sampler(
            self,
            data_source: DataIdMultiDataset,
            batch_size: int,
            task_filter: Optional[Container[str]] = None):
        return BatchOneTaskTemperatureProportionalSampler(
            data_source, batch_size, self.batches_per_epoch, 1, self.max_contribution, task_filter)


@dataclasses.dataclass(frozen=True)
class BatchOneTaskTemperatureProportionalSamplerFactory(BatchOneTaskSamplerFactory):
    temperature: float
    max_contribution: Optional[int] = None

    def make_sampler(
            self,
            data_source: DataIdMultiDataset,
            batch_size: int,
            task_filter: Optional[Container[str]] = None):
        return BatchOneTaskTemperatureProportionalSampler(
            data_source, batch_size, self.batches_per_epoch, self.temperature, self.max_contribution, task_filter)


class BatchOneTaskSampler(torch.utils.data.Sampler):

    def __init__(
            self,
            data_source: DataIdMultiDataset,
            batch_size: int,
            batches_per_epoch: int,
            task_filter: Optional[Container[str]] = None):
        super().__init__(data_source)
        self.task_indices = data_source.task_indices()
        if task_filter is not None:
            self.task_indices = type(self.task_indices)(
                (k, self.task_indices[k]) for k in self.task_indices
                if k in task_filter or data_source.response_data_kind(k) in task_filter)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(batches_per_epoch, int) or isinstance(batches_per_epoch, bool) or \
                batches_per_epoch <= 0:
            raise ValueError("batches_per_epoch should be a positive integer value, "
                             "but got batch_per_epoch={}".format(batches_per_epoch))
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch

    def __iter__(self):
        idx = 0
        while self.batches_per_epoch <= 0 or idx < self.batches_per_epoch:
            task = self._sample_task()
            task_sample = np.random.randint(0, len(self.task_indices[task]), self.batch_size)
            batch = list()
            for i in task_sample:
                if len(batch) + len(self.task_indices[task][i]) > self.batch_size:
                    if len(batch) == 0:
                        batch.append(self.task_indices[task][i])
                    break
                batch.append(self.task_indices[task][i])
            yield np.concatenate(batch)
            idx += 1

    def _sample_task(self) -> str:
        task_keys = [task for task in self.task_indices]
        return np.random.choice(task_keys)

    def true_div_len(self):
        return self.batches_per_epoch

    def __len__(self):
        return self.batches_per_epoch


class BatchOneTaskTemperatureProportionalSampler(BatchOneTaskSampler):

    def __init__(
            self,
            data_source: DataIdMultiDataset,
            batch_size: int,
            batches_per_epoch: int,
            temperature: float,
            max_contribution: Optional[int] = None,
            task_filter: Optional[Container[str]] = None):
        super().__init__(data_source, batch_size, batches_per_epoch, task_filter)
        self._task_keys = [task for task in self.task_indices]
        self._sample_rates = np.array(list(len(self.task_indices[k]) for k in self._task_keys))
        if max_contribution is not None:
            self._sample_rates = np.minimum(self._sample_rates, max_contribution)
        self._sample_rates = np.power(self._sample_rates / np.sum(self._sample_rates), 1 / temperature)
        self._sample_rates = self._sample_rates / np.sum(self._sample_rates)

    def _sample_task(self):
        return np.random.choice(self._task_keys, p=self._sample_rates)


class BatchOneTaskSequentialSamplerFactory(SamplerFactory):

    @classmethod
    def is_batch_sampler(cls):
        return True

    @classmethod
    def is_one_task_at_a_time_sampler(cls):
        return True

    def make_sampler(
            self,
            data_source: DataIdMultiDataset,
            batch_size: int,
            task_filter: Optional[Container[str]] = None):
        return BatchOneTaskSequentialSampler(
            data_source, batch_size, task_filter)


class BatchOneTaskRandomSamplerFactory(SamplerFactory):

    @classmethod
    def is_one_task_at_a_time_sampler(cls):
        return True

    @classmethod
    def is_batch_sampler(cls):
        return True

    def make_sampler(
            self,
            data_source: DataIdMultiDataset,
            batch_size: int,
            task_filter: Optional[Container[str]] = None):
        return BatchOneTaskRandomSampler(data_source, batch_size, task_filter)


class BatchOneTaskRandomSampler(torch.utils.data.Sampler):

    def __init__(self, data_source: DataIdMultiDataset, batch_size, task_filter=None):
        super().__init__(data_source)
        self.task_indices = data_source.task_indices()
        if task_filter is not None:
            self.task_indices = type(self.task_indices)(
                (k, self.task_indices[k]) for k in self.task_indices
                if k in task_filter or data_source.response_data_kind(k) in task_filter)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        self.batch_size = batch_size

    def __iter__(self):
        batches = list()
        for task in self.task_indices:
            task_sample = np.random.permutation(len(self.task_indices[task]))
            batch = list()
            for i in task_sample:
                if len(batch) + len(self.task_indices[task][i]) > self.batch_size:
                    if len(batch) > 0:
                        batches.append(np.concatenate(batch))
                    batch = list()
                # if a single multipart item > batch_count, we just make a batch that is larger than batch size
                # so no check here
                batch.append(self.task_indices[task][i])
            if len(batch) > 0:
                batches.append(np.concatenate(batch))
        batches = np.random.permutation(batches)
        for batch in batches:
            yield batch

    def true_div_len(self):
        return (
            sum(sum(len(task_item_list) for task_item_list in self.task_indices[task]) for task in self.task_indices)
            / self.batch_size)

    def __len__(self):
        return int(self.true_div_len())


class RandomSamplerFactory(SamplerFactory):

    @classmethod
    def is_batch_sampler(cls):
        return False

    @classmethod
    def is_one_task_at_a_time_sampler(cls):
        return False

    def make_sampler(
            self,
            data_source: DataIdMultiDataset,
            batch_size: int,
            task_filter: Optional[Container[str]] = None):
        if task_filter is not None:
            raise ValueError('{} does not support task_filter'.format(type(self)))
        return torch.utils.data.RandomSampler(data_source)


class BatchOneTaskSequentialSampler(torch.utils.data.Sampler):

    def __init__(self, data_source: DataIdMultiDataset, batch_size, task_filter=None):
        super().__init__(data_source)
        self.task_indices = data_source.task_indices()
        if task_filter is not None:
            self.task_indices = type(self.task_indices)(
                (k, self.task_indices[k]) for k in self.task_indices
                if k in task_filter or data_source.response_data_kind(k) in task_filter)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        self.batch_size = batch_size

    def __iter__(self):
        for task in self.task_indices:
            task_sample = np.arange(len(self.task_indices[task]))
            batch = list()
            for i in task_sample:
                if len(batch) + len(self.task_indices[task][i]) > self.batch_size:
                    yield np.concatenate(batch)
                    batch = list()
                # if a single multipart item > batch_count, we just make a batch that is larger than batch size
                # so no check here
                batch.append(self.task_indices[task][i])
            if len(batch) > 0:
                yield np.concatenate(batch)

    def true_div_len(self):
        return (
            sum(sum(len(task_item_list) for task_item_list in self.task_indices[task]) for task in self.task_indices)
            / self.batch_size)

    def __len__(self):
        return int(self.true_div_len())
