import dataclasses
from collections import OrderedDict
from typing import Optional, Container, Iterable

import numpy as np
import torch
import torch.utils.data

from .data_id_dataset import DataIdDataset
from .data_id_multidataset import DataIdMultiDataset


__all__ = [
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
    uncertainty_log_sigma_squared_field: Optional[str] = None
    num_uncertainty_warmup_steps: int = 0

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
        return BatchOneTaskSampler(
            data_source, batch_size, self.batches_per_epoch, task_filter,
            self.uncertainty_log_sigma_squared_field, self.num_uncertainty_warmup_steps)


@dataclasses.dataclass(frozen=True)
class BatchOneTaskProportionalSamplerFactory(BatchOneTaskSamplerFactory):
    max_contribution: Optional[int] = None

    def make_sampler(
            self,
            data_source: DataIdMultiDataset,
            batch_size: int,
            task_filter: Optional[Container[str]] = None):
        return BatchOneTaskTemperatureProportionalSampler(
            data_source, batch_size, self.batches_per_epoch, 1, self.max_contribution, task_filter,
            self.uncertainty_log_sigma_squared_field, self.num_uncertainty_warmup_steps)


@dataclasses.dataclass(frozen=True)
class BatchOneTaskTemperatureProportionalSamplerFactory(BatchOneTaskSamplerFactory):
    temperature: float = 1
    max_contribution: Optional[int] = None

    def make_sampler(
            self,
            data_source: DataIdMultiDataset,
            batch_size: int,
            task_filter: Optional[Container[str]] = None):
        return BatchOneTaskTemperatureProportionalSampler(
            data_source, batch_size, self.batches_per_epoch, self.temperature, self.max_contribution, task_filter,
            self.uncertainty_log_sigma_squared_field, self.num_uncertainty_warmup_steps)


def _gather_uncertainty(data_source, uncertainty_log_sigma_squared_field, batch, predictions, loss_handlers):
    loss_dict = dict((loss.field, loss) for loss in loss_handlers)
    if DataIdDataset.response_id_field not in batch:
        raise ValueError('{} not present in batch'.format(DataIdDataset.response_id_field))
    if uncertainty_log_sigma_squared_field not in predictions:
        raise ValueError('log_sigma_squared not in predictions: {}'.format(uncertainty_log_sigma_squared_field))
    assert (len(batch[DataIdDataset.response_id_field])
            == len(predictions[uncertainty_log_sigma_squared_field]))
    batch_uncertainty_weights = dict()
    for response_id, log_sigma_squared in zip(
            batch[DataIdDataset.response_id_field], predictions[uncertainty_log_sigma_squared_field]):
        task_key = data_source.response_field_for_id(response_id.item())
        if task_key not in loss_dict:
            raise ValueError('loss not found for {}'.format(task_key))
        uncertainty_weight = loss_dict[task_key].uncertainty_weight() * np.exp(log_sigma_squared.item())
        if task_key not in batch_uncertainty_weights:
            batch_uncertainty_weights[task_key] = [uncertainty_weight]
        else:
            batch_uncertainty_weights[task_key].append(uncertainty_weight)
    return batch_uncertainty_weights


class BatchOneTaskSampler(torch.utils.data.Sampler):

    def __init__(
            self,
            data_source: DataIdMultiDataset,
            batch_size: int,
            batches_per_epoch: int,
            task_filter: Optional[Container[str]] = None,
            uncertainty_log_sigma_squared_field: Optional[str] = None,
            num_uncertainty_warmup_steps: int = 0):
        super().__init__(data_source)
        self._data_source = data_source
        self._uncertainty_log_sigma_squared_field = uncertainty_log_sigma_squared_field
        self._num_uncertainty_warmup_steps = num_uncertainty_warmup_steps
        self._num_uncertainty_steps = 0
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
        self._task_keys = [task for task in self.task_indices]
        self._sample_rates = np.ones(len(self._task_keys))
        self._sample_rates /= np.sum(self._sample_rates)
        self._task_uncertainty = np.ones_like(self._sample_rates)
        self._effective_rates = self._task_uncertainty * self._sample_rates
        self._effective_rates = self._effective_rates / np.sum(self._effective_rates)

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

    def _sample_task(self):
        return np.random.choice(self._task_keys, p=self._effective_rates)

    def update(self, batch, predictions, loss_handlers):
        if self._uncertainty_log_sigma_squared_field is None:
            return
        if self._num_uncertainty_steps >= self._num_uncertainty_warmup_steps:
            task_uncertainty = _gather_uncertainty(
                self._data_source, self._uncertainty_log_sigma_squared_field, batch, predictions, loss_handlers)
            for index_task, task_key in enumerate(self._task_keys):
                if task_key in task_uncertainty:
                    self._task_uncertainty[index_task] = np.mean(task_uncertainty[task_key])
            self._effective_rates = self._task_uncertainty * self._sample_rates
            self._effective_rates = self._effective_rates / np.sum(self._effective_rates)
        self._num_uncertainty_steps += 1

    def task_uncertainty_weights(self):
        return OrderedDict(zip(self._task_keys, self._task_uncertainty))

    def update_from(self, other_samplers: Iterable['BatchOneTaskSampler']):
        task_uncertainty = dict()
        for sampler in other_samplers:
            task_uncertainty.update(sampler.task_uncertainty_weights())
        for index_task, task_key in enumerate(self._task_keys):
            if task_key in task_uncertainty:
                self._task_uncertainty[index_task] = task_uncertainty[task_key]
        self._effective_rates = self._task_uncertainty * self._sample_rates
        self._effective_rates = self._effective_rates / np.sum(self._effective_rates)

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
            task_filter: Optional[Container[str]] = None,
            uncertainty_log_sigma_squared_field: Optional[str] = None,
            num_uncertainty_warmup_steps: int = 0):
        super().__init__(
            data_source, batch_size, batches_per_epoch, task_filter,
            uncertainty_log_sigma_squared_field, num_uncertainty_warmup_steps)
        self._task_keys = [task for task in self.task_indices]
        self._sample_rates = np.array(list(len(self.task_indices[k]) for k in self._task_keys))
        if max_contribution is not None:
            self._sample_rates = np.minimum(self._sample_rates, max_contribution)
        self._sample_rates = np.power(self._sample_rates / np.sum(self._sample_rates), 1 / temperature)
        self._sample_rates = self._sample_rates / np.sum(self._sample_rates)
        self._task_uncertainty = np.ones_like(self._sample_rates)
        self._effective_rates = self._task_uncertainty * self._sample_rates
        self._effective_rates = self._effective_rates / np.sum(self._effective_rates)


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
