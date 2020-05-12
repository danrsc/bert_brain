import dataclasses
from collections import OrderedDict
from typing import Optional, Container, Iterable, Callable, Mapping
import logging

import numpy as np
from scipy.special import softmax as scipy_softmax
import torch
import torch.utils.data

from .data_id_dataset import DataIdDataset
from .data_id_multidataset import DataIdMultiDataset


logger = logging.getLogger(__name__)


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
    'RandomSamplerFactory',
    'BatchOneTaskTaskPermutedSamplerFactory',
    'BatchOneTaskTaskPermutedSampler',
    'BatchOneTaskManualWeightSamplerFactory',
    'BatchOneTaskManualWeightSampler',
    'BatchOneTaskEvalSampler',
    'BatchOneTaskMultiDifferentiableDataSelectionSampler',
    'BatchOneTaskMultiDifferentiableDataSelectionSamplerFactory']


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
    max_contribution: Optional[float] = None

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
    max_contribution: Optional[float] = None

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


@dataclasses.dataclass(frozen=True)
class BatchOneTaskManualWeightSamplerFactory(SamplerFactory):
    batches_per_epoch: int
    weight_fn: Callable[[Mapping[str, float]], Mapping[str, float]]
    uncertainty_log_sigma_squared_field: Optional[str] = None
    num_uncertainty_warmup_steps: int = 0

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
        return BatchOneTaskManualWeightSampler(
            data_source, self.weight_fn, batch_size, self.batches_per_epoch, task_filter,
            self.uncertainty_log_sigma_squared_field, self.num_uncertainty_warmup_steps)


class BatchOneTaskManualWeightSampler(BatchOneTaskSampler):

    def __init__(
            self,
            data_source: DataIdMultiDataset,
            weight_fn: Callable[[Mapping[str, float]], Mapping[str, float]],
            batch_size: int,
            batches_per_epoch: int,
            task_filter: Optional[Container[str]] = None,
            uncertainty_log_sigma_squared_field: Optional[str] = None,
            num_uncertainty_warmup_steps: int = 0):
        super().__init__(
            data_source, batch_size, batches_per_epoch, task_filter, uncertainty_log_sigma_squared_field,
            num_uncertainty_warmup_steps)

        proportions = np.array(list(len(self.task_indices[k]) for k in self._task_keys))
        proportions = proportions / np.sum(proportions)
        sample_rates = dict(weight_fn(dict(zip(self._task_keys, proportions))))
        if len(sample_rates) != len(self._task_keys):
            raise ValueError('A sample rate must be returned for every task')
        self._sample_rates = list()
        for key in self._task_keys:
            if key not in sample_rates:
                raise ValueError('A sample rate must be returned for every task')
            self._sample_rates.append(sample_rates[key])
        self._sample_rates = np.array(self._sample_rates)
        self._sample_rates /= np.sum(self._sample_rates)
        self._task_uncertainty = np.ones_like(self._sample_rates)
        self._effective_rates = self._task_uncertainty * self._sample_rates
        self._effective_rates = self._effective_rates / np.sum(self._effective_rates)


class BatchOneTaskTemperatureProportionalSampler(BatchOneTaskSampler):

    def __init__(
            self,
            data_source: DataIdMultiDataset,
            batch_size: int,
            batches_per_epoch: int,
            temperature: float,
            max_contribution: Optional[float] = None,
            task_filter: Optional[Container[str]] = None,
            uncertainty_log_sigma_squared_field: Optional[str] = None,
            num_uncertainty_warmup_steps: int = 0):
        super().__init__(
            data_source, batch_size, batches_per_epoch, task_filter,
            uncertainty_log_sigma_squared_field, num_uncertainty_warmup_steps)
        self._task_keys = [task for task in self.task_indices]
        self._sample_rates = np.array(list(len(self.task_indices[k]) for k in self._task_keys))
        self._sample_rates = self._sample_rates / np.sum(self._sample_rates)
        if max_contribution is not None:
            self._sample_rates = np.minimum(self._sample_rates, max_contribution)
            self._sample_rates = self._sample_rates / np.sum(self._sample_rates)
        self._sample_rates = np.power(self._sample_rates, 1 / temperature)
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


@dataclasses.dataclass(frozen=True)
class RandomSamplerFactory(SamplerFactory):
    replacement: bool = False
    num_samples: Optional[int] = None

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
        return torch.utils.data.RandomSampler(data_source, self.replacement, self.num_samples)


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


@dataclasses.dataclass(frozen=True)
class BatchOneTaskTaskPermutedSamplerFactory(SamplerFactory):
    batches_per_task_per_epoch: int = 1

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
        return BatchOneTaskTaskPermutedSampler(data_source, batch_size, self.batches_per_task_per_epoch, task_filter)


class BatchOneTaskTaskPermutedSampler(torch.utils.data.Sampler):

    def __init__(
            self,
            data_source: DataIdMultiDataset,
            batch_size: int,
            batches_per_task_per_epoch: int,
            task_filter: Optional[Container[str]] = None):
        super().__init__(data_source)
        self._data_source = data_source
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
        self.batches_per_task_per_epoch = batches_per_task_per_epoch
        self._task_keys = [task for task in self.task_indices]

    def __iter__(self):
        for _ in range(self.batches_per_task_per_epoch):
            for task in np.random.permutation(list(self.task_indices)):
                task_sample = np.random.randint(0, len(self.task_indices[task]), self.batch_size)
                batch = list()
                for i in task_sample:
                    if len(batch) + len(self.task_indices[task][i]) > self.batch_size:
                        if len(batch) == 0:
                            batch.append(self.task_indices[task][i])
                        break
                    batch.append(self.task_indices[task][i])
                yield np.concatenate(batch)

    def true_div_len(self):
        return self.batches_per_task_per_epoch * len(self.task_indices)

    def __len__(self):
        return int(self.true_div_len())


class BatchOneTaskEvalSampler(torch.utils.data.Sampler):

    def __init__(
            self,
            data_source: DataIdMultiDataset,
            batch_size: int,
            max_samples_per_task: int,
            task_filter: Optional[Container[str]] = None):
        super().__init__(data_source)
        self._data_source = data_source
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
        self.max_samples_per_task = max_samples_per_task
        self._task_keys = [task for task in self.task_indices]

    def __iter__(self):
        for task in self.task_indices:
            if len(self.task_indices[task]) < self.max_samples_per_task:
                task_sample = np.arange(len(self.task_indices[task]))
            else:
                task_sample = np.random.permutation(len(self.task_indices[task]))
            batch = list()
            count = 0
            for i in task_sample:
                if len(batch) + len(self.task_indices[task][i]) > self.batch_size:
                    yield np.concatenate(batch)
                    count += len(batch)
                    batch = list()
                    if count >= self.max_samples_per_task:
                        break
                # if a single multipart item > batch_count, we just make a batch that is larger than batch size
                # so no check here
                batch.append(self.task_indices[task][i])
            if len(batch) > 0:
                yield np.concatenate(batch)

    def true_div_len(self):
        return (
            sum(min(sum(len(task_item_list) for task_item_list in self.task_indices[task]), self.max_samples_per_task)
                for task in self.task_indices)
            / self.batch_size)

    def __len__(self):
        return int(self.true_div_len())


@dataclasses.dataclass(frozen=True)
class BatchOneTaskMultiDifferentiableDataSelectionSamplerFactory(SamplerFactory):
    batches_per_epoch: int
    update_frequency_in_batches: int = 100
    initial_sample_rate_proportional_temperature: int = 1
    learning_rate: Optional[float] = None
    preferences: Optional[Mapping[str, float]] = None

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
        return BatchOneTaskMultiDifferentiableDataSelectionSampler(
            data_source, batch_size, self.batches_per_epoch, task_filter,
            self.initial_sample_rate_proportional_temperature,
            self.learning_rate,
            self.preferences)


class BatchOneTaskMultiDifferentiableDataSelectionSampler(torch.utils.data.Sampler):

    def __init__(
            self,
            data_source: DataIdMultiDataset,
            batch_size: int,
            batches_per_epoch: int,
            task_filter: Optional[Container[str]] = None,
            initial_sample_rate_proportional_temperature: int = 1,
            learning_rate: Optional[float] = None,
            preferences: Optional[Mapping[str, float]] = None):
        super().__init__(data_source)
        self._data_source = data_source
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
        self.learning_rate = learning_rate
        self._task_keys = [task for task in self.task_indices]
        if preferences is not None:
            self._preferences = OrderedDict()
            for k in self._task_keys:
                if k in preferences:
                    self._preferences[k] = preferences[k]
                elif data_source.response_data_kind(k) in preferences:
                    self._preferences[k] = preferences[data_source.response_data_kind(k)]
                elif data_source.data_set_key_for_field(k) in preferences:
                    self._preferences[k] = preferences[data_source.data_set_key_for_field(k)]
                else:
                    self._preferences[k] = 1
        else:
            self._preferences = OrderedDict((k, 1) for k in self._task_keys)
        self._temperature = 1
        self._sample_rate_logits = np.zeros(len(self._task_keys))

        sample_rates = np.array(list(len(self.task_indices[k]) for k in self._task_keys))
        sample_rates = sample_rates / np.sum(sample_rates)
        sample_rates = np.power(sample_rates, 1 / initial_sample_rate_proportional_temperature)
        sample_rates = torch.tensor(sample_rates / np.sum(sample_rates))

        # temporary parameter to initialize to target sample rates
        sample_rate_logits = torch.nn.Parameter(torch.tensor(self._sample_rate_logits), requires_grad=True)

        optimizer = torch.optim.SGD(params=[sample_rate_logits], lr=1)

        loss = None
        while loss is None or loss > 0.000001:
            sm_rates = torch.nn.functional.softmax(sample_rate_logits, dim=-1)
            loss = torch.nn.functional.mse_loss(sm_rates, sample_rates)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss = loss.item()

        self._sample_rates = torch.nn.functional.softmax(sample_rate_logits, dim=-1).detach().numpy()
        self._sample_rate_logits = sample_rate_logits.detach().numpy()

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
        return np.random.choice(self._task_keys, p=self._sample_rates)

    def update_rate_logits(self, rate_logits):
        if len(rate_logits) != len(self._task_keys):
            raise ValueError('rate_logits length does not match task keys')
        logits = list()
        for task_key in self._task_keys:
            if task_key not in rate_logits:
                raise ValueError('rate_logits is missing task_key: {}'.format(task_key))
            logits.append(rate_logits[task_key])
        self._sample_rate_logits = np.array(logits)
        new_sample_rates = scipy_softmax(self._sample_rate_logits)
        diff = self._sample_rates - new_sample_rates
        if self._temperature != 1:
            new_sample_rates = np.power(new_sample_rates, 1 / self._temperature)
            new_sample_rates = new_sample_rates / np.sum(new_sample_rates)
        self._sample_rates = new_sample_rates
        logger.info('new sample rates: {}'.format(self._sample_rates))
        logger.info('sample rates diff: {}'.format(diff))

    @property
    def temperature(self):
        return self._temperature

    def change_temperature(self, temperature):
        self._temperature = temperature

    def rate_logits(self):
        return OrderedDict(zip(self._task_keys, self._sample_rate_logits))

    def preferences(self):
        return OrderedDict(self._preferences)

    def true_div_len(self):
        return self.batches_per_epoch

    def __len__(self):
        return self.batches_per_epoch
