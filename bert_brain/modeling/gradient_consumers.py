from typing import Mapping
from collections import OrderedDict

import numpy as np
import torch

from ..data_sets import DataIdDataset


__all__ = ['GradientContainer', 'MovingAverageGradientSimilarity', 'GradientCounter']


class GradientContainer:

    def __init__(self, use_cpu=False, store_norm=False, keep_only_nonzero=True):
        self._use_cpu = use_cpu
        self._store_norm = store_norm
        self._keep_only_nonzero = keep_only_nonzero
        self._gradient_container = dict()
        self._norm_container = dict() if self._store_norm else None

    @property
    def use_cpu(self):
        return self._use_cpu

    @property
    def store_norm(self):
        return self._store_norm

    @property
    def keep_only_nonzero(self):
        return self._keep_only_nonzero

    def add(self, gradient: Mapping[str, torch.tensor], batch, data_set):
        for key in gradient:
            if self.use_cpu:
                g = gradient[key].cpu()
            else:
                g = gradient[key].clone()
            keep = True
            if self.keep_only_nonzero or self.store_norm:
                norm = torch.sqrt(torch.sum(g ** 2))
                if self.keep_only_nonzero:
                    keep = norm > 0
                if self.store_norm:
                    if key not in self._norm_container:
                        self._norm_container[key] = list()
                    self._norm_container[key].append(norm)
            if keep:
                if key not in self._gradient_container:
                    self._gradient_container[key] = list()
                self._gradient_container[key].append(g)

    def gradients(self, key):
        return self._gradient_container[key]

    def norms(self, key):
        if self._norm_container is None:
            raise ValueError('Cannot request norms when store_norm=False')
        return self._norm_container[key]

    def clear(self):
        self._gradient_container = dict()
        self._norm_container = dict() if self.store_norm else None

    def __contains__(self, item):
        return item in self._gradient_container


class MovingAverageGradientSimilarity:

    def __init__(
            self,
            tasks,
            store_only_train=True,
            sim_only_meta_train=True,
            preferences=None,
            decay=0.9,
            device=None):
        self._last_index_task = None
        self._last_gradient = None
        self._last_sq_norm = None
        self._task_to_index = dict((t, i) for i, t in enumerate(tasks))
        self._store_only_train = store_only_train
        self._sim_only_meta_train = sim_only_meta_train
        self._task_similarity = torch.full((len(self._task_to_index), len(self._task_to_index)), np.nan, device=device)
        if preferences is not None:
            self._task_preferences = torch.unsqueeze(
                torch.tensor(list(preferences[t] for t in tasks), device=device), 0)
        else:
            self._task_preferences = torch.ones((1, len(self._task_to_index)), device=device)
        self._decay = decay

    def clear(self):
        self._last_index_task = None
        self._last_gradient = None
        self._last_sq_norm = None
        self._task_similarity = torch.full(
            (len(self._task_to_index), len(self._task_to_index)), np.nan, device=self._task_similarity.device)

    @property
    def decay(self):
        return self._decay

    def add(self, gradient: Mapping[str, torch.tensor], batch, data_set):
        task = None
        if torch.all(batch[DataIdDataset.response_id_field] == batch[DataIdDataset.response_id_field][0]):
            task = data_set.response_field_for_id(batch[DataIdDataset.response_id_field][0])
        if task is None:
            raise ValueError('task not present')
        if task not in self._task_to_index:
            raise ValueError('Unknown task: {}'.format(task))
        index_task = self._task_to_index[task]
        current_gradient = dict()
        current_sq_norm = dict()
        dot = 0
        total_norm_current = 0
        total_norm_last = 0

        should_update_sim = data_set.which == 'meta_train' or not self._sim_only_meta_train

        for key in gradient:
            g = torch.reshape(gradient[key].clone(), (-1,))
            sq_norm = torch.sum(g ** 2)
            if sq_norm > 0:
                current_gradient[key] = g
                current_sq_norm[key] = sq_norm
                if should_update_sim and self._last_index_task is not None:
                    if key in self._last_gradient:
                        dot += torch.matmul(current_gradient[key], self._last_gradient[key])
                        total_norm_current += current_sq_norm[key]
                        total_norm_last += self._last_sq_norm[key]
        if should_update_sim and self._last_index_task is not None and total_norm_current > 0 and total_norm_last > 0:
            similarity = dot / (torch.sqrt(total_norm_current) * torch.sqrt(total_norm_last))
            if torch.isnan(self._task_similarity[index_task, self._last_index_task]):
                self._task_similarity[index_task, self._last_index_task] = similarity
            else:
                self._task_similarity[index_task, self._last_index_task] = (
                        self.decay * self._task_similarity[index_task, self._last_index_task]
                        + (1 - self.decay) * similarity)
        if data_set.which == 'train' or not self._store_only_train:
            self._last_index_task = index_task
            self._last_gradient = current_gradient
            self._last_sq_norm = current_sq_norm

    def mean_task_similarity(self, replace_nan_with_mean=True):
        indicator_finite = ~torch.isnan(self._task_similarity)
        if not torch.any(indicator_finite):
            mean = torch.full(len(self._task_similarity), np.nan)
        else:
            if replace_nan_with_mean:
                mean = torch.mean(
                    torch.where(
                        indicator_finite,
                        self._task_preferences * self._task_similarity,
                        self._task_preferences * torch.mean(self._task_similarity[indicator_finite])),
                    dim=1)
            else:
                mean = torch.sum(
                    torch.where(indicator_finite, self._task_preferences * self._task_similarity, 0), dim=1)
                mean = mean / torch.sum(indicator_finite, dim=1)
        return OrderedDict((t, mean[self._task_to_index[t]]) for t in self._task_to_index)

    def pairwise_similarities(self):
        for task1 in self._task_to_index:
            for task2 in self._task_to_index:
                yield task1, task2, self._task_similarity[self._task_to_index[task1], self._task_to_index[task2]]


class GradientCounter:

    def __init__(self):
        self._gradients = dict()
        self._counts = dict()

    def add(self, gradient: Mapping[str, torch.tensor], batch, data_set):
        for key in gradient:
            if key not in self._gradients:
                self._gradients[key] = gradient[key].clone()
                self._counts[key] = 1
            elif torch.any(gradient[key] != self._gradients[key]):
                self._gradients[key] = gradient[key].clone()
                self._counts[key] = self._counts[key] + 1

    def count(self, key):
        if key not in self._counts:
            return 0
        return self._counts[key]

    def clear(self):
        self._gradients = dict()
        self._counts = dict()
