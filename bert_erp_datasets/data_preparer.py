import warnings
from concurrent.futures import ProcessPoolExecutor
from collections import OrderedDict
from functools import partial
from itertools import groupby, chain
from dataclasses import dataclass, replace
from typing import List, Mapping, Any, Optional
import numpy as np
from scipy.stats import boxcox
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.decomposition import PCA

from bert_erp_tokenization import FieldSpec
from bert_erp_common import FrozenCopyOfDict


__all__ = [
    'DataPreparer',
    'PreparedData',
    'split_data',
    'PreprocessBoxcox',
    'PreprocessLog',
    'PreprocessDetrend',
    'PreprocessDiscretize',
    'PreprocessBaseline',
    'PreprocessSequenceStandardize',
    'PreprocessDiff',
    'PreprocessStandardize',
    'PreprocessMakeBinary',
    'PreprocessNanMean',
    'PreprocessPCA',
    'PreprocessNanGeometricMean',
    'PreprocessClip',
    'PreprocessGaussianBlur',
    'PreprocessCompress',
    'PreprocessMany']


def split_data(to_split, test_proportion, validation_of_train_proportion, shuffle=True, random_state=None):
    from sklearn.model_selection import train_test_split

    if test_proportion > 0:
        idx_train, idx_test = train_test_split(
            np.arange(len(to_split)), test_size=test_proportion, shuffle=shuffle, random_state=random_state)
    else:
        idx_train = np.arange(len(to_split))
        idx_test = []

    if validation_of_train_proportion > 0:
        idx_train, idx_validation = train_test_split(
            idx_train, test_size=validation_of_train_proportion, shuffle=shuffle, random_state=random_state)
    else:
        idx_validation = []

    train = [to_split[i] for i in idx_train]
    validation = [to_split[i] for i in idx_validation]
    test = [to_split[i] for i in idx_test]
    return train, validation, test


def _polyfit(item):
    key, data = item
    x = np.arange(len(data))
    indicator_finite = np.isfinite(data)
    x = x[indicator_finite]
    data = data[indicator_finite]
    p = np.polyfit(x, data, 1)
    return key, p


def _boxcox(item):
    key, data = item
    data = data[np.logical_not(np.isnan(data))]
    _, transform_lambda = boxcox(data)
    return key, transform_lambda


def _group_responses(response_groups, indices=None):
    unique_groups = np.unique(response_groups)
    indicators_groups = [response_groups == g for g in unique_groups]

    # assign the indices within the group
    if indices is not None:
        new_indices = np.full(len(response_groups), -1)
        for indicator_group in indicators_groups:
            new_indices[indicator_group] = np.arange(int(np.sum(indicator_group)))
        indicator_invalid = np.logical_or(indices < 0, indices >= len(response_groups))
        invalid_train_indices = indices[indicator_invalid]
        indices[indicator_invalid] = 0
        indices = new_indices[indices]
        indices[indicator_invalid] = invalid_train_indices

    for indicator_group in indicators_groups:
        if indices is not None:
            yield indicator_group, indices[indicator_group]
        else:
            yield indicator_group


class _PreprocessBase:

    def __init__(self, data_key_whitelist=None, data_key_blacklist=None):
        self.data_key_whitelist, self.data_key_blacklist = data_key_whitelist, data_key_blacklist

    def _iterate_data_columns(self, data, indices=None, response_groups=None):
        """
        Helper function for applying a pre-processing step in parallel to each "column" of data. Used in conjunction
        with _collect_data_column_results
        Args:
            data: The data to pre-process
            indices: Optional. Indices into the data (usually from examples.
                If specified, takes the data at these indices only (along axis=0)
            response_groups: Optional. Groups specified along axis=0. If present, data is grouped according to these
                groups, and then collected by group. For example, this can be used to apply a step within a
                recording block rather than across all blocks for detrending.
        Returns:
            A generator (key, data_column) pairs if response_groups is None, or (key, index_group, data_column) triples
            when response_groups is specified
        """
        for k in self._iterate_data_keys(data):
            d = data[k]
            if response_groups is not None:
                if indices is not None:
                    grouped_d = list()
                    for indicator_group, indices_group in _group_responses(response_groups, indices):
                        group_d = d[indicator_group]
                        grouped_d.append(group_d[indices_group])
                else:
                    grouped_d = [d[indicator_group] for indicator_group in _group_responses(response_groups)]
            else:
                if indices is not None:
                    d = d[indices]
                grouped_d = [d]
            for index_group, d in enumerate(grouped_d):
                d = np.reshape(d, (d.shape[0], -1))
                for i in range(d.shape[1]):
                    if response_groups is not None:
                        yield k, index_group, d[:, i]
                    else:
                        yield k, d[:, i]

    @classmethod
    def _collect_data_column_results(cls, items, response_groups=None):
        """
        Helper function for applying a pre-processing step in parallel to each "column" of data. Used in conjunction
        with _iterate_data_columns

        Args:
            items: The results of the parallel-processing
            response_groups: Optional. Used to properly filter the groups after the parallel execution
        Returns:
            A generator of (key, data_columns) if response_groups is not specified or
            (key, response_group: List[(indicator_group, data_columns)]) if response_groups is specified
        """
        indicators_groups = None
        if response_groups is not None:
            unique_groups = np.unique(response_groups)
            indicators_groups = [response_groups == g for g in unique_groups]

        for k, group in groupby(items, lambda itm: itm[0]):
            if response_groups is not None:
                response_grouped = list()
                for index_group, response_group in groupby(group, lambda itm: itm[1]):
                    response_grouped.append((indicators_groups[index_group], [item[2] for item in response_group]))
                yield k, response_grouped
            else:
                yield k, [item[1] for item in group]

    def _get_data_indices(self, examples):
        return [ex.data_ids for ex in examples]

    def _parallel_data_column_map(self, fn, data, compute_on_examples=None, response_groups=None, **kwargs):
        indices = None
        if compute_on_examples is not None:
            indices = np.concatenate(self._get_data_indices(compute_on_examples))
            indices = indices[indices >= 0]

        modified = dict()
        with ProcessPoolExecutor() as ex:
            for key, items in self._collect_data_column_results(
                    ex.map(fn, self._iterate_data_columns(data, indices, response_groups)), response_groups):
                modified[key] = self._parallel_apply(data[key], key, items, response_groups, **kwargs)
        return modified

    def _map(self, data):
        return dict((k, self._apply(data[k])) for k in self._iterate_data_keys(data))

    def _iterate_data_keys(self, data):
        for k in data:
            if self.data_key_whitelist is not None and k not in self.data_key_whitelist:
                continue
            if self.data_key_blacklist is not None and k in self.data_key_blacklist:
                continue
            yield k

    def _apply(self, data_k):
        raise RuntimeError('Unexpected call to _apply in {}'.format(type(self)))

    def _parallel_apply(self, data_k, key, items, response_groups, **kwargs):
        raise RuntimeError('Unexpected call to _parallel_apply in {}'.format(type(self)))


class _PreprocessStopWordAwareBase(_PreprocessBase):

    def __init__(self, data_key_whitelist=None, data_key_blacklist=None, stop_mode=None):
        super().__init__(data_key_whitelist, data_key_blacklist)
        self.stop_mode = stop_mode

    def _get_data_indices(self, examples):
        result = list()
        for ex in examples:
            data_indices = ex.data_ids
            if self.stop_mode == 'content':
                data_indices = np.where(ex.input_is_stop, -1, data_indices)
            elif self.stop_mode == 'stop':
                data_indices = np.where(ex.input_is_stop, data_indices, -1)
            elif self.stop_mode is not None:
                raise ValueError('Unable to understand stop_mode: {}'.format(self.stop_mode))
            result.append(data_indices)
        return result


class PreprocessBoxcox(_PreprocessStopWordAwareBase):

    def _parallel_apply(self, data_k, key, transform_lambdas, response_groups, **kwargs):
        assert(response_groups is None)
        from scipy.stats import boxcox
        data_k = np.copy(data_k)
        shape = data_k.shape
        data_k = np.reshape(data_k, (shape[0], -1))
        for idx, transform_lambda in enumerate(transform_lambdas):
            indicator_valid = np.logical_not(np.isnan(data_k[:, idx]))
            data_k[indicator_valid, idx] = boxcox(data_k[indicator_valid, idx], transform_lambda)
        return np.reshape(data_k, shape)

    def __call__(self, loaded_data_tuple, metadata):
        modified = self._parallel_data_column_map(
            _boxcox, loaded_data_tuple.data, compute_on_examples=loaded_data_tuple.train)
        return replace(loaded_data_tuple, data=FrozenCopyOfDict.replace(loaded_data_tuple.data, modified))


class PreprocessLog(_PreprocessBase):

    def __init__(self, min_value=-20, data_key_whitelist=None, data_key_blacklist=None):
        super().__init__(data_key_whitelist, data_key_blacklist)
        self.min_value = min_value

    def _apply(self, data_k):
        isnan = np.isnan(data_k)
        data_k = np.where(isnan, 1, data_k)
        if np.any(np.less(data_k, 0)):
            raise ValueError('Values must be >= 0')
        data_k = np.log(np.maximum(data_k, np.power(np.e, self.min_value)))
        return np.where(isnan, np.nan, data_k)

    def __call__(self, loaded_data_tuple, metadata):
        modified = self._map(loaded_data_tuple.data)
        return replace(loaded_data_tuple, data=FrozenCopyOfDict.replace(loaded_data_tuple.data, modified))


class PreprocessDetrend(_PreprocessStopWordAwareBase):

    def __init__(
            self, data_key_whitelist=None, data_key_blacklist=None, stop_mode=None, metadata_response_group_by=None):
        super().__init__(data_key_whitelist, data_key_blacklist, stop_mode)
        self.metadata_response_group_by = metadata_response_group_by

    @staticmethod
    def _apply_coefficients(data, coefficients):
        data = np.copy(data)
        shape = data.shape
        data = np.reshape(data, (data.shape[0], -1))
        for idx, p in enumerate(coefficients):
            line = p[0] * np.arange(len(data[:, idx])) + p[1]
            data[:, idx] -= line
        return np.reshape(data, shape)

    def __call__(self, loaded_data_tuple, metadata):

        train_indices = np.concatenate(self._get_data_indices(loaded_data_tuple.train))
        valid_train_indices = train_indices[train_indices >= 0]
        if not np.all(np.diff(valid_train_indices) > 0):
            raise ValueError('expected passages and data to be in order')

        response_groups = None
        if self.metadata_response_group_by is not None:
            if metadata is None or self.metadata_response_group_by not in metadata:
                raise ValueError('metadata_response_group_by not found in metadata')
            response_groups = metadata[self.metadata_response_group_by]

        modified = dict()
        with ProcessPoolExecutor() as ex:
            for k, items in self._collect_data_column_results(
                    ex.map(
                        _polyfit,
                        self._iterate_data_columns(loaded_data_tuple.data, valid_train_indices, response_groups)),
                    response_groups):
                if response_groups is None:
                    modified[k] = PreprocessDetrend._apply_coefficients(loaded_data_tuple.data[k], items)
                else:
                    modified[k] = np.copy(loaded_data_tuple.data[k])
                    for indicator_group, coefficients in items:
                        modified[k][indicator_group] = PreprocessDetrend._apply_coefficients(
                            modified[k][indicator_group], coefficients)

        return replace(loaded_data_tuple, data=FrozenCopyOfDict.replace(loaded_data_tuple.data, modified))


class PreprocessDiscretize(_PreprocessBase):

    def __init__(self, bins=10, range=None, use_one_hot=True, data_key_whitelist=None, data_key_blacklist=None):
        super().__init__(data_key_whitelist, data_key_blacklist)
        self.bins = bins
        self.range = range
        self.use_one_hot = use_one_hot

    def _apply(self, data):
        bin_edges = np.histogram_bin_edges(data, self.bins, range)
        if np.isscalar(self.bins):
            bin_edges = bin_edges[1:]
        data = np.digitize(data, bin_edges, right=True)
        if self.use_one_hot:
            one_hot = np.zeros(data.shape + (len(bin_edges) + 1,), data.dtype)
            one_hot = np.reshape(one_hot, (-1, one_hot.shape[-1]))
            for idx, bin in enumerate(np.reshape(data, -1)):
                one_hot[idx, bin] = 1
            data = np.reshape(one_hot, data.shape + (one_hot.shape[-1],))
        return data

    def __call__(self, loaded_data_tuple, metadata):
        modified = self._map(loaded_data_tuple.data)
        return replace(loaded_data_tuple, data=FrozenCopyOfDict.replace(loaded_data_tuple.data, modified))


class PreprocessBaseline(_PreprocessBase):

    def __init__(self, num_baseline, data_key_whitelist=None, data_key_blacklist=None):
        """
        Computes a running mean of the last num_baseline non-nan values and subtracts this running mean
        from the data. This completely ignores example boundaries. Validation/test examples are removed if the baselines
        from those examples would overlap with train examples
        """
        super().__init__(data_key_whitelist, data_key_blacklist)
        self.num_baseline = num_baseline

    def _parallel_apply(self, data_k, key, baselines, response_groups, **kwargs):
        assert(response_groups is None)
        baselines = np.reshape(
            np.concatenate([np.expand_dims(b, axis=1) for b in baselines], axis=1),
            data_k.shape)

        return data_k - baselines

    def __call__(self, loaded_data_tuple, metadata):
        max_train_index = None
        if loaded_data_tuple.train is not None:
            for t in loaded_data_tuple.train:
                max_t = np.nanmax(t.data_ids)
                if max_train_index is None or max_t > max_train_index:
                    max_train_index = max_t

        if max_train_index is not None and loaded_data_tuple.validation is not None:
            clean_validation = list()
            for v in loaded_data_tuple.validation:
                v_indices = np.where(v.data_ids < 0, np.nan, v.data_ids)
                min_v = np.nanmin(v_indices)
                if min_v is None or min_v - self.num_baseline > max_train_index:
                    clean_validation.append(v)

            loaded_data_tuple = replace(loaded_data_tuple, validation=clean_validation)

        if max_train_index is not None and loaded_data_tuple.test is not None:
            clean_test = list()
            for t in loaded_data_tuple.test:
                t_indices = np.where(t.data_ids < 0, np.nan, t.data_ids)
                min_t = np.nanmin(t_indices)
                if min_t is None or min_t - self.num_baseline > max_train_index:
                    clean_test.append(t)
            loaded_data_tuple = replace(loaded_data_tuple, test=clean_test)

        fn = partial(_compute_baseline, num_baseline=self.num_baseline)
        modified = self._parallel_data_column_map(fn, loaded_data_tuple.data)
        return replace(loaded_data_tuple, data=FrozenCopyOfDict.replace(loaded_data_tuple.data, modified))


def _compute_baseline(item, num_baseline):

    key, vals = item

    def _iterate():
        current = list()
        for v in vals:
            if not np.isnan(v):
                current.append(v)
                if len(current) > num_baseline:
                    current = current[1:]
            yield np.mean(current) if len(current) > 0 else 0

    return key, np.array(list(_iterate()))


class PreprocessSequenceStandardize(_PreprocessStopWordAwareBase):

    def __call__(self, loaded_data_tuple, metadata):

        modified = dict(
            (k, np.copy(loaded_data_tuple.data[k])) for k in self._iterate_data_keys(loaded_data_tuple.data))

        for t in chain(loaded_data_tuple.train, loaded_data_tuple.validation, loaded_data_tuple.test):

            data_indices = t.data_ids
            compute_indices = data_indices
            if self.stop_mode == 'content':
                compute_indices = np.where(t.input_is_stop, -1, compute_indices)
            elif self.stop_mode == 'stop':
                compute_indices = np.where(t.input_is_stop, compute_indices, -1)
            elif self.stop_mode is not None:
                raise ValueError('Unable to understand stop_mode: {}'.format(self.stop_mode))

            data_indices = data_indices[data_indices >= 0]
            compute_indices = compute_indices[compute_indices >= 0]
            for k in self._iterate_data_keys(loaded_data_tuple.data):
                compute = modified[k][compute_indices]
                compute = np.reshape(compute, (compute.shape[0], -1))
                mean = np.mean(compute, axis=0, keepdims=True)
                std = np.std(compute, axis=0, keepdims=True)

                data = modified[k][data_indices]
                data_shape = data.shape
                data = np.reshape(data, (data.shape[0], -1))
                data = (data - mean) / std
                modified[k][data_indices] = np.reshape(data, data_shape)

        return replace(loaded_data_tuple, data=FrozenCopyOfDict.replace(loaded_data_tuple.data, modified))


class PreprocessDiff(_PreprocessBase):

    def __init__(self, fill_value=0., data_key_whitelist=None, data_key_blacklist=None):
        # should this support stop_mode? much more complicated, unlikely to use
        super().__init__(data_key_whitelist, data_key_blacklist)
        self.fill_value = fill_value

    def _apply(self, data_k):
        padding = np.full((1,) + data_k.shape[1:], self.fill_value, data_k.dtype)
        return np.concatenate([padding, np.diff(data_k, axis=0)])

    def __call__(self, loaded_data_tuple, metadata):
        modified = self._map(loaded_data_tuple.data)
        return replace(loaded_data_tuple, data=FrozenCopyOfDict.replace(loaded_data_tuple.data, modified))


class PreprocessPCA(_PreprocessStopWordAwareBase):

    def __init__(
            self,
            feature_axis=1,  # features with respect to PCA, e.g. subjects
            data_key_whitelist=None,
            data_key_blacklist=None,
            stop_mode=None):
        super().__init__(data_key_whitelist, data_key_blacklist, stop_mode)
        self.feature_axis = feature_axis

    def __call__(self, loaded_data_tuple, metadata):

        train_indices = np.concatenate(self._get_data_indices(loaded_data_tuple.train))
        train_indices = train_indices[train_indices >= 0]

        modified = dict()
        for k in self._iterate_data_keys(loaded_data_tuple.data):

            all_values = loaded_data_tuple.data[k]
            # -> (samples, ..., features)
            all_values = np.moveaxis(all_values, self.feature_axis, -1)
            result_shape = all_values.shape[:-1]
            # -> (samples, task, features)
            all_values = np.reshape(
                all_values,
                (all_values.shape[0], int(np.prod(all_values.shape[1:-1]))), all_values.shape[-1])
            # -> (task, samples, features)
            all_values = np.transpose(all_values, (1, 0, 2))
            result = list()
            for current in all_values:
                pca = PCA(n_components=1)
                train_values = current[train_indices]
                pca.fit(train_values)
                result.append(pca.transform(current))

            # -> (task, samples, 1)
            result = np.array(result)
            # -> (samples, task, 1)
            result = np.transpose(result, (1, 0, 2))
            # -> (samples, ...)
            result = np.reshape(result, result_shape)

            modified[k] = result

        return replace(loaded_data_tuple, data=FrozenCopyOfDict.replace(loaded_data_tuple.data, modified))


class PreprocessStandardize(_PreprocessStopWordAwareBase):

    def __init__(
            self,
            average_axis=1,
            data_key_whitelist=None,
            data_key_blacklist=None,
            stop_mode=None,
            metadata_response_group_by=None):
        super().__init__(data_key_whitelist, data_key_blacklist, stop_mode)
        self.average_axis = average_axis
        self.metadata_response_group_by = metadata_response_group_by

    def _standardize(self, data, valid_train_indices):

        valid_train_values = data[valid_train_indices]

        pre_average_mean = np.nanmean(valid_train_values, axis=0, keepdims=True)
        pre_average_std = np.nanstd(valid_train_values, axis=0, keepdims=True)

        transformed_data = (data - pre_average_mean) / pre_average_std

        if self.average_axis is not None:
            standardized_train_values = (valid_train_values - pre_average_mean) / pre_average_std
            with warnings.catch_warnings():
                # catch 'Mean of emtpy slice' warning here
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                standardized_train_values = np.nanmean(standardized_train_values, axis=self.average_axis)
                transformed_data = np.nanmean(transformed_data, axis=self.average_axis)
            post_average_mean = np.nanmean(standardized_train_values, axis=0, keepdims=True)
            post_average_std = np.nanstd(standardized_train_values, axis=0, keepdims=True)
            transformed_data = (transformed_data - post_average_mean) / post_average_std

        return transformed_data

    def __call__(self, loaded_data_tuple, metadata):

        train_indices = np.concatenate(self._get_data_indices(loaded_data_tuple.train))

        modified = dict()
        for k in self._iterate_data_keys(loaded_data_tuple.data):
            if self.metadata_response_group_by is None:
                valid_train_indices = train_indices[train_indices >= 0]
                modified[k] = self._standardize(loaded_data_tuple.data[k], valid_train_indices)
            else:
                if metadata is None or self.metadata_response_group_by not in metadata:
                    raise ValueError('metadata_response_group_by not found: {}'.format(self.metadata_response_group_by))
                modified[k] = np.copy(loaded_data_tuple.data[k])
                for indicator_group, group_train_indices in _group_responses(
                        metadata[self.metadata_response_group_by], train_indices):
                    valid_train_indices = group_train_indices[group_train_indices >= 0]
                    modified[k][indicator_group] = self._standardize(modified[k][indicator_group], valid_train_indices)

        return replace(loaded_data_tuple, data=FrozenCopyOfDict.replace(loaded_data_tuple.data, modified))


class PreprocessMakeBinary(_PreprocessBase):

    def __init__(self, threshold, dtype=np.int32, data_key_whitelist=None, data_key_blacklist=None):
        super().__init__(data_key_whitelist, data_key_blacklist)
        self.threshold = threshold
        self.dtype = dtype

    def _apply(self, data_k):
        indicator_nan = np.isnan(data_k)
        safe_compare = np.where(indicator_nan, self.threshold - 1, data_k)
        indicator = np.logical_and(np.logical_not(indicator_nan), safe_compare >= self.threshold)
        if self.dtype == np.bool_ or self.dtype == bool:
            return indicator
        return np.where(indicator, np.ones(data_k.shape, self.dtype), np.zeros(data_k.shape, self.dtype))

    def __call__(self, loaded_data_tuple, metadata):
        modified = self._map(loaded_data_tuple.data)
        return replace(loaded_data_tuple, data=FrozenCopyOfDict.replace(loaded_data_tuple.data, modified))


class PreprocessNanMean(_PreprocessBase):

    def __init__(self, data_key_whitelist=None, data_key_blacklist=None, axis=1):
        super().__init__(data_key_whitelist, data_key_blacklist)
        self.axis = axis

    def _apply(self, data_k):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(data_k, self.axis)

    def __call__(self, loaded_data_tuple, metadata):
        modified = self._map(loaded_data_tuple.data)
        return replace(loaded_data_tuple, data=FrozenCopyOfDict.replace(loaded_data_tuple.data, modified))


class PreprocessClip(_PreprocessBase):

    def __init__(
            self, minimum=None, maximum=None, value_beyond_min=None, value_beyond_max=None,
            data_key_whitelist=None, data_key_blacklist=None):
        super().__init__(data_key_whitelist, data_key_blacklist)
        self.value_beyond_min = value_beyond_min
        self.value_beyond_max = value_beyond_max
        self.min = minimum
        self.max = maximum

    def _apply(self, data_k):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            if self.min is not None:
                if self.value_beyond_min is not None:
                    data_k = np.where(data_k < self.min, self.value_beyond_min, data_k)
                else:
                    data_k = np.maximum(self.min, data_k)
            if self.max is not None:
                if self.value_beyond_max is not None:
                    data_k = np.where(data_k > self.max, self.value_beyond_max, data_k)
                else:
                    data_k = np.minimum(self.max, data_k)
            return data_k

    def __call__(self, loaded_data_tuple, metadata):
        modified = self._map(loaded_data_tuple.data)
        return replace(loaded_data_tuple, data=FrozenCopyOfDict.replace(loaded_data_tuple.data, modified))


class PreprocessNanGeometricMean(_PreprocessBase):

    def __init__(self, data_key_whitelist=None, data_key_blacklist=None, axis=1):
        super().__init__(data_key_whitelist, data_key_blacklist)
        self.axis = axis

    def _apply(self, data_k):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            return np.exp(np.nanmean(np.log(data_k), axis=self.axis))

    def __call__(self, loaded_data_tuple, metadata):
        modified = self._map(loaded_data_tuple.data)
        return replace(loaded_data_tuple, data=FrozenCopyOfDict.replace(loaded_data_tuple.data, modified))


class PreprocessGaussianBlur(_PreprocessBase):

    def __init__(
            self, sigma=1, data_key_whitelist=None, data_key_blacklist=None, axis=1,
            order=0, mode='reflect', cval=0.0, truncate=4.0):
        """
        This is meant to blue over a non-example axis, e.g. spatially in fMRI
        """
        super().__init__(data_key_whitelist, data_key_blacklist)
        self.sigma = sigma
        self.axis = axis
        self.order = order
        self.mode = mode
        self.cval = cval
        self.truncate = truncate

    def _apply(self, data_k):
        return gaussian_filter1d(
            data_k, sigma=self.sigma, axis=self.axis, order=self.order, mode=self.mode, truncate=self.truncate)

    def __call__(self, loaded_data_tuple, metadata):
        modified = self._map(loaded_data_tuple.data)
        return replace(loaded_data_tuple, data=FrozenCopyOfDict.replace(loaded_data_tuple.data, modified))


class PreprocessCompress(_PreprocessBase):

    def __init__(self, metadata_condition_name, compress_axis=1, data_key_whitelist=None, data_key_blacklist=None):
        super().__init__(data_key_whitelist, data_key_blacklist)
        self.metadata_condition_name, self.compress_axis = metadata_condition_name, compress_axis

    def __call__(self, loaded_data_tuple, metadata):
        if metadata is None or self.metadata_condition_name not in metadata:
            raise ValueError('Unable to find metadata_condition_name: {}'.format(self.metadata_condition_name))
        condition = metadata[self.metadata_condition_name]
        modified = dict(
            (k, np.compress(condition, loaded_data_tuple.data[k], axis=self.compress_axis))
            for k in self._iterate_data_keys(loaded_data_tuple.data))
        return replace(loaded_data_tuple, data=FrozenCopyOfDict.replace(loaded_data_tuple.data, modified))


class PreprocessMany:

    def __init__(self, *steps):
        self.steps = steps

    def __call__(self, loaded_data_tuple, metadata):
        for step in self.steps:
            loaded_data_tuple = step(loaded_data_tuple, metadata)

        return loaded_data_tuple


@dataclass(frozen=True)
class PreparedData:
    train: Optional[List[Mapping[str, Any]]] = None
    validation: Optional[List[Mapping[str, Any]]] = None
    test: Optional[List[Mapping[str, Any]]] = None
    data: Optional[Mapping[str, Any]] = None
    field_specs: Optional[Mapping[str, FieldSpec]] = None


class DataPreparer(object):

    def __init__(self, seed, preprocess_dict):
        self._seed = seed
        self._random_state = dict()
        self._prepared_cache = dict()
        self._preprocess_dict = dict(preprocess_dict)

    def prepare(self, raw_data_dict):
        result = OrderedDict()
        metadata = OrderedDict()

        for k in raw_data_dict:
            metadata[k] = raw_data_dict[k].metadata
            if raw_data_dict[k].is_pre_split:
                result[k] = PreparedData(
                    raw_data_dict[k].input_examples,
                    raw_data_dict[k].validation_input_examples,
                    raw_data_dict[k].test_input_examples,
                    FrozenCopyOfDict(raw_data_dict[k].response_data),
                    field_specs=raw_data_dict[k].field_specs)
            elif raw_data_dict[k].split_function is not None:
                if k not in self._random_state:
                    self._random_state[k] = np.random.RandomState(self._seed)
                train_input_examples, validation_input_examples, test_input_examples = raw_data_dict[k].split_function(
                    raw_data_dict[k], self._random_state[k])
                result[k] = PreparedData(
                    train_input_examples, validation_input_examples, test_input_examples,
                    FrozenCopyOfDict(raw_data_dict[k].response_data),
                    field_specs=raw_data_dict[k].field_specs)
            else:
                if k not in self._random_state:
                    self._random_state[k] = np.random.RandomState(self._seed)
                train_input_examples, validation_input_examples, test_input_examples = split_data(
                    raw_data_dict[k].input_examples,
                    raw_data_dict[k].test_proportion,
                    raw_data_dict[k].validation_proportion_of_train,
                    random_state=self._random_state[k])
                loaded_data_tuple = PreparedData(
                    train_input_examples,
                    validation_input_examples,
                    test_input_examples,
                    FrozenCopyOfDict(raw_data_dict[k].response_data),
                    field_specs=raw_data_dict[k].field_specs)
                result[k] = loaded_data_tuple

        for k in result:
            if k in self._preprocess_dict:
                if raw_data_dict[k].is_pre_split:
                    if k not in self._prepared_cache:
                        self._prepared_cache[k] = self._preprocess_dict[k](result[k], metadata[k])
                    result[k] = self._prepared_cache[k]
                else:
                    result[k] = self._preprocess_dict[k](result[k], metadata[k])

        return result
