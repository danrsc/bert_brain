import warnings
from concurrent.futures import ProcessPoolExecutor
from collections import OrderedDict
from functools import partial
from itertools import groupby, chain
from dataclasses import dataclass, replace
from typing import List, Mapping, Any, Optional
from bert_erp_common import FrozenCopyOfDict
import numpy as np
from scipy.stats import boxcox
from sklearn.decomposition import PCA


__all__ = [
    'DataPreparer',
    'LoadedDataTuple',
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


def _iterate_data_keys(data, data_key_whitelist, data_key_blacklist):
    for k in data:
        if data_key_whitelist is not None and k not in data_key_whitelist:
            continue
        if data_key_blacklist is not None and k in data_key_blacklist:
            continue
        yield k


def _iterate_data_columns(data, indices, data_key_whitelist, data_key_blacklist):
    for k in _iterate_data_keys(data, data_key_whitelist, data_key_blacklist):
        d = data[k]
        if indices is not None:
            d = d[indices]
        d = np.reshape(d, (d.shape[0], -1))
        for i in range(d.shape[1]):
            yield k, d[:, i]


def _collect_data_column_results(items):
    for k, group in groupby(items, lambda item: item[0]):
        yield k, [item[1] for item in group]


class PreprocessBoxcox:

    def __init__(
            self, data_key_whitelist=None, data_key_blacklist=None, stop_mode=None):
        self.data_key_whitelist = data_key_whitelist
        self.data_key_blacklist = data_key_blacklist
        self.stop_mode = stop_mode

    def __call__(self, loaded_data_tuple):
        from scipy.stats import boxcox
        train_indices = np.concatenate(
            _get_data_indices(loaded_data_tuple.train, self.stop_mode))
        train_indices = train_indices[train_indices >= 0]

        modified = dict()
        with ProcessPoolExecutor() as ex:
            for k, transform_lambdas in _collect_data_column_results(ex.map(
                    _boxcox,
                    _iterate_data_columns(
                        loaded_data_tuple.data, train_indices, self.data_key_whitelist, self.data_key_blacklist))):
                data_k = np.copy(loaded_data_tuple.data[k])
                data_k = np.reshape(data_k, (loaded_data_tuple.data[k].shape[0], -1))
                for idx, transform_lambda in enumerate(transform_lambdas):
                    indicator_valid = np.logical_not(np.isnan(data_k[:, idx]))
                    data_k[indicator_valid, idx] = boxcox(data_k[indicator_valid, idx], transform_lambda)
                modified[k] = np.reshape(data_k, loaded_data_tuple.data[k].shape)

        return replace(loaded_data_tuple, data=FrozenCopyOfDict.replace(loaded_data_tuple.data, modified))


class PreprocessLog:

    def __init__(self, min_value=-20, data_key_whitelist=None, data_key_blacklist=None):
        self.min_value = min_value
        self.data_key_whitelist = data_key_whitelist
        self.data_key_blacklist = data_key_blacklist

    def __call__(self, loaded_data_tuple):
        modified = dict()
        for k in _iterate_data_keys(loaded_data_tuple.data, self.data_key_whitelist, self.data_key_blacklist):

            isnan = np.isnan(loaded_data_tuple.data[k])
            data_k = np.where(isnan, 1, loaded_data_tuple.data[k])
            if np.any(np.less(data_k, 0)):
                raise ValueError('Values must be >= 0')
            data_k = np.log(np.maximum(data_k, np.power(np.e, self.min_value)))
            modified[k] = np.where(isnan, np.nan, data_k)
        return replace(loaded_data_tuple, data=FrozenCopyOfDict.replace(loaded_data_tuple.data, modified))


class PreprocessDetrend:

    def __init__(
            self, data_key_whitelist=None, data_key_blacklist=None, stop_mode=None):
        self.data_key_whitelist = data_key_whitelist
        self.data_key_blacklist = data_key_blacklist
        self.stop_mode = stop_mode

    def __call__(self, loaded_data_tuple):
        train_indices = np.concatenate(
            _get_data_indices(loaded_data_tuple.train, self.stop_mode))
        train_indices = train_indices[train_indices >= 0]
        if not np.all(np.diff(train_indices) > 0):
            raise ValueError('expected passages and data to be in order')

        modified = dict()
        with ProcessPoolExecutor() as ex:
            for k, coefficients in _collect_data_column_results(ex.map(
                    _polyfit,
                    _iterate_data_columns(
                        loaded_data_tuple.data, train_indices, self.data_key_whitelist, self.data_key_blacklist))):
                data_k = np.copy(loaded_data_tuple.data[k])
                data_k = np.reshape(data_k, (loaded_data_tuple.data[k].shape[0], -1))
                for idx, p in enumerate(coefficients):
                    line = p[0] * np.arange(len(data_k[:, idx])) + p[1]
                    data_k[:, idx] -= line

                modified[k] = np.reshape(data_k, loaded_data_tuple.data[k].shape)

        return replace(loaded_data_tuple, data=FrozenCopyOfDict.replace(loaded_data_tuple.data, modified))


class PreprocessDiscretize:

    def __init__(self, bins=10, range=None, use_one_hot=True, data_key_whitelist=None, data_key_blacklist=None):
        self.bins = bins
        self.range = range
        self.use_one_hot = use_one_hot
        self.data_key_whitelist = data_key_whitelist
        self.data_key_blacklist = data_key_blacklist

    def __call__(self, loaded_data_tuple):
        modified = dict()
        for k in _iterate_data_keys(loaded_data_tuple.data, self.data_key_whitelist, self.data_key_blacklist):
            data = loaded_data_tuple.data[k]
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
            modified[k] = data

        return replace(loaded_data_tuple, data=FrozenCopyOfDict.replace(loaded_data_tuple.data, modified))


class PreprocessBaseline:

    def __init__(self, num_baseline, data_key_whitelist=None, data_key_blacklist=None):
        self.num_baseline = num_baseline
        self.data_key_whitelist = data_key_whitelist
        self.data_key_blacklist = data_key_blacklist

    def __call__(self, loaded_data_tuple):
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

        modified = dict()
        fn = partial(_compute_baseline, num_baseline=self.num_baseline)
        with ProcessPoolExecutor() as ex:
            for k, baselines in _collect_data_column_results(ex.map(
                    fn,
                    _iterate_data_columns(
                        loaded_data_tuple.data, None, self.data_key_whitelist, self.data_key_blacklist))):

                baselines = np.reshape(
                    np.concatenate([np.expand_dims(b, axis=1) for b in baselines], axis=1),
                    loaded_data_tuple.data[k].shape)

                modified[k] = loaded_data_tuple.data[k] - baselines

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


class PreprocessSequenceStandardize:

    def __init__(
            self, data_key_whitelist=None, data_key_blacklist=None, stop_mode=None):
        self.data_key_whitelist, self.data_key_blacklist, self.stop_mode = (
            data_key_whitelist, data_key_blacklist, stop_mode)

    def __call__(self, loaded_data_tuple):
        modified = dict()
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
            for k in _iterate_data_keys(loaded_data_tuple.data, self.data_key_whitelist, self.data_key_blacklist):
                modified[k] = np.copy(loaded_data_tuple.data[k])
                compute = modified[k]
                data = modified[k]
                compute = compute[compute_indices]
                data_shape = data.shape
                compute = np.reshape(compute, (compute.shape[0], -1))
                data = np.reshape(data, (data.shape[0], -1))
                mean = np.mean(compute, axis=0, keepdims=True)
                std = np.std(compute, axis=0, keepdims=True)
                data = (data - mean) / std
                modified[k][data_indices] = np.reshape(data, data_shape)

        return replace(loaded_data_tuple, data=FrozenCopyOfDict.replace(loaded_data_tuple.data, modified))


class PreprocessDiff:

    def __init__(self, fill_value=0., data_key_whitelist=None, data_key_blacklist=None):
        # should this support stop_mode? much more complicated, unlikely to use
        self.fill_value = fill_value
        self.data_key_whitelist = data_key_whitelist
        self.data_key_blacklist = data_key_blacklist

    def __call__(self, loaded_data_tuple):
        modified = dict()
        for k in _iterate_data_keys(loaded_data_tuple.data, self.data_key_whitelist, self.data_key_blacklist):
            padding = np.full(
                (1,) + loaded_data_tuple.data[k].shape[1:], self.fill_value, loaded_data_tuple.data[k].dtype)
            modified[k] = np.concatenate([padding, np.diff(loaded_data_tuple.data[k], axis=0)])

        return replace(loaded_data_tuple, data=FrozenCopyOfDict.replace(loaded_data_tuple.data, modified))


class PreprocessPCA:

    def __init__(
            self,
            feature_axis=1,
            data_key_whitelist=None,
            data_key_blacklist=None,
            stop_mode=None):
        (self.feature_axis, self.data_key_whitelist, self.data_key_blacklist, self.stop_mode) = (
            feature_axis, data_key_whitelist, data_key_blacklist, stop_mode)

    def __call__(self, loaded_data_tuple):

        train_indices = np.concatenate(
            _get_data_indices(loaded_data_tuple.train, self.stop_mode), axis=0)
        valid_train_indices = train_indices[train_indices >= 0]

        modified = dict()
        for k in _iterate_data_keys(loaded_data_tuple.data, self.data_key_whitelist, self.data_key_blacklist):

            all_values = loaded_data_tuple.data[k]
            # -> (samples, ..., features)
            all_values = np.moveaxis(all_values, self.feature_axis, -1)
            result_shape = all_values.shape[:-1]
            # -> (samples, task, features)
            all_values = np.reshape(
                all_values,
                (all_values.shape[0], max(1, int(np.prod(all_values.shape[1:-1]))), all_values.shape[-1]))
            # -> (task, samples, features)
            all_values = np.transpose(all_values, (1, 0, 2))
            result = list()
            for current in all_values:
                pca = PCA(n_components=1)
                train_values = current[valid_train_indices]
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


def _get_data_indices(examples, stop_mode):
    result = list()
    for ex in examples:
        data_indices = ex.data_ids
        if stop_mode == 'content':
            data_indices = np.where(ex.input_is_stop, -1, data_indices)
        elif stop_mode == 'stop':
            data_indices = np.where(ex.input_is_stop, data_indices, -1)
        elif stop_mode is not None:
            raise ValueError('Unable to understand stop_mode: {}'.format(stop_mode))
        result.append(data_indices)
    return result


class PreprocessStandardize:

    def __init__(
            self,
            average_axis=1,
            data_key_whitelist=None,
            data_key_blacklist=None,
            stop_mode=None):
        (self.average_axis, self.data_key_whitelist, self.data_key_blacklist, self.stop_mode) = (
            average_axis, data_key_whitelist, data_key_blacklist, stop_mode)

    def __call__(self, loaded_data_tuple):

        train_indices = np.concatenate(
            _get_data_indices(loaded_data_tuple.train, self.stop_mode), axis=0)
        valid_train_indices = train_indices[train_indices >= 0]

        modified = dict()
        for k in _iterate_data_keys(loaded_data_tuple.data, self.data_key_whitelist, self.data_key_blacklist):

            valid_train_values = loaded_data_tuple.data[k][valid_train_indices]

            pre_average_mean = np.nanmean(valid_train_values, axis=0, keepdims=True)
            pre_average_std = np.nanstd(valid_train_values, axis=0, keepdims=True)

            transformed_data = (loaded_data_tuple.data[k] - pre_average_mean) / pre_average_std

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

            modified[k] = transformed_data

        return replace(loaded_data_tuple, data=FrozenCopyOfDict.replace(loaded_data_tuple.data, modified))


class PreprocessMakeBinary:

    def __init__(self, threshold, dtype=np.int32, data_key_whitelist=None, data_key_blacklist=None):
        self.threshold = threshold
        self.dtype = dtype
        self.data_key_whitelist = data_key_whitelist
        self.data_key_blacklist = data_key_blacklist

    def __call__(self, loaded_data_tuple):
        modified = dict()
        for k in _iterate_data_keys(loaded_data_tuple.data, self.data_key_whitelist, self.data_key_blacklist):
            safe_compare = np.where(np.isnan(loaded_data_tuple.data[k]), self.threshold - 1, loaded_data_tuple.data[k])
            indicator = np.where(
                np.isnan(loaded_data_tuple.data[k]), False, safe_compare >= self.threshold)
            if self.dtype == np.bool_ or self.dtype == bool:
                modified[k] = indicator
            else:
                modified[k] = np.where(
                    indicator,
                    np.ones(loaded_data_tuple.data[k].shape, self.dtype),
                    np.zeros(loaded_data_tuple.data[k].shape, self.dtype))
        return replace(loaded_data_tuple, data=FrozenCopyOfDict.replace(loaded_data_tuple.data, modified))


class PreprocessNanMean:

    def __init__(self, data_key_whitelist=None, data_key_blacklist=None, axis=1):
        self.data_key_whitelist = data_key_whitelist
        self.data_key_blacklist = data_key_blacklist
        self.axis = axis

    def __call__(self, loaded_data_tuple):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            modified = dict((k, np.nanmean(loaded_data_tuple.data[k], axis=self.axis)) for k in _iterate_data_keys(
                loaded_data_tuple.data, self.data_key_whitelist, self.data_key_blacklist))
        return replace(loaded_data_tuple, data=FrozenCopyOfDict.replace(loaded_data_tuple.data, modified))


class PreprocessClip:

    def __init__(self, minimum=None, maximum=None, data_key_whitelist=None, data_key_blacklist=None):
        self.data_key_whitelist = data_key_whitelist
        self.data_key_blacklist = data_key_blacklist
        self.min = minimum
        self.max = maximum

    def _clip(self, x):
        if self.min is not None:
            x = np.maximum(self.min, x)
        if self.max is not None:
            x = np.minimum(self.max, x)
        return x

    def __call__(self, loaded_data_tuple):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            modified = dict((k, self._clip(loaded_data_tuple.data[k])) for k in _iterate_data_keys(
                loaded_data_tuple.data, self.data_key_whitelist, self.data_key_blacklist))
        return replace(loaded_data_tuple, data=FrozenCopyOfDict.replace(loaded_data_tuple.data, modified))


class PreprocessNanGeometricMean:

    def __init__(self, data_key_whitelist=None, data_key_blacklist=None, axis=1):
        self.data_key_whitelist = data_key_whitelist
        self.data_key_blacklist = data_key_blacklist
        self.axis = axis

    def _geom_mean(self, x):
        return np.exp(np.nanmean(np.log(x), axis=self.axis))

    def __call__(self, loaded_data_tuple):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            modified = dict((k, self._geom_mean(loaded_data_tuple.data[k])) for k in _iterate_data_keys(
                loaded_data_tuple.data, self.data_key_whitelist, self.data_key_blacklist))
        return replace(loaded_data_tuple, data=FrozenCopyOfDict.replace(loaded_data_tuple.data, modified))


class PreprocessMany:

    def __init__(self, *steps):
        self.steps = steps

    def __call__(self, loaded_data_tuple):
        for step in self.steps:
            loaded_data_tuple = step(loaded_data_tuple)

        return loaded_data_tuple


@dataclass(frozen=True)
class LoadedDataTuple:
    train: Optional[List[Mapping[str, Any]]] = None
    validation: Optional[List[Mapping[str, Any]]] = None
    test: Optional[List[Mapping[str, Any]]] = None
    data: Optional[Mapping[str, Any]] = None


class DataPreparer(object):

    def __init__(self, seed, preprocess_dict):
        self._seed = seed
        self._random_state = dict()
        self._prepared_cache = dict()
        self._preprocess_dict = dict(preprocess_dict)

    def prepare(self, raw_data_dict):
        result = OrderedDict()

        for k in raw_data_dict:
            if raw_data_dict[k].is_pre_split:
                result[k] = LoadedDataTuple(
                    raw_data_dict[k].input_examples,
                    raw_data_dict[k].validation_input_examples,
                    raw_data_dict[k].test_input_examples,
                    FrozenCopyOfDict(raw_data_dict[k].response_data))
            else:
                if k not in self._random_state:
                    self._random_state[k] = np.random.RandomState(self._seed)
                train_input_examples, validation_input_examples, test_input_examples = split_data(
                    raw_data_dict[k].input_examples,
                    raw_data_dict[k].test_proportion,
                    raw_data_dict[k].validation_proportion_of_train,
                    random_state=self._random_state[k])
                loaded_data_tuple = LoadedDataTuple(
                    train_input_examples,
                    validation_input_examples,
                    test_input_examples,
                    FrozenCopyOfDict(raw_data_dict[k].response_data))
                result[k] = loaded_data_tuple

        for k in result:
            if k in self._preprocess_dict:
                if raw_data_dict[k].is_pre_split:
                    if k not in self._prepared_cache:
                        self._prepared_cache[k] = self._preprocess_dict[k](result[k])
                    result[k] = self._prepared_cache[k]
                else:
                    result[k] = self._preprocess_dict[k](result[k])

        return result
