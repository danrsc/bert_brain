import warnings
from types import MappingProxyType
from concurrent.futures import ProcessPoolExecutor
from collections import OrderedDict
from itertools import chain
from dataclasses import dataclass, replace
from typing import Mapping, Optional, Sequence, Callable, Tuple
import numpy as np
from scipy.stats import boxcox
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.decomposition import PCA

from .input_features import InputFeatures, FieldSpec, RawData, KindData

__all__ = [
    'DataPreparer',
    'PreparedData',
    'PreparedDataView',
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


def _fit_boxcox(item):
    data, indicator_train = item
    if indicator_train is not None:
        data = data[indicator_train]
    data = data[np.logical_not(np.isnan(data))]
    _, transform_lambda = boxcox(data)
    return transform_lambda


def _apply_boxcox(data, transform_lambdas):
    data = np.copy(data)
    for idx, transform_lambda in enumerate(transform_lambdas):
        indicator_valid = np.logical_not(np.isnan(data[:, idx]))
        data[indicator_valid, idx] = boxcox(data[indicator_valid, idx], transform_lambda)
    return data


def _indicator_from_examples(data_size, examples, stop_mode=None):
    result = np.full(data_size, False)
    for ex in examples:
        data_ids = ex.data_ids
        if stop_mode is not None:
            if stop_mode == 'content':
                data_ids = np.where(ex.is_stop, -1, data_ids)
            elif stop_mode == 'stop':
                data_ids = np.where(ex.is_stop, data_ids, -1)
            else:
                raise ValueError('Unknown value for stop_mode: {}'.format(stop_mode))
        data_ids = data_ids[data_ids >= 0]
        result[data_ids] = True
    return result


def _parallel_column_map(fit_fn, apply_fn, data, indicator_fit=None):
    shape = data.shape
    data = np.reshape(data, (data.shape[0], -1))
    with ProcessPoolExecutor() as ex:
        fit_result = ex.map(fit_fn, [(data[i, :], indicator_fit) for i in data.shape[1]])
    assert(len(fit_result) == data.shape[1])
    data = apply_fn(data, fit_result)
    return np.reshape(data, shape)


class PreprocessBoxcox:

    def __call__(self, loaded_data_tuple, metadata):
        data = _parallel_column_map(_fit_boxcox, _apply_boxcox, loaded_data_tuple.data)
        return replace(loaded_data_tuple, data=data)


class PreprocessLog:

    def __init__(self, min_value: float = -20.):
        self.min_value = min_value

    def __call__(self, loaded_data_tuple, metadata):
        isnan = np.isnan(loaded_data_tuple.data)
        data = np.where(isnan, 1, loaded_data_tuple.data)
        if np.any(np.less(data, 0)):
            raise ValueError('Values must be >= 0')
        data = np.log(np.maximum(data, np.power(np.e, self.min_value)))
        return replace(loaded_data_tuple, data=np.where(isnan, np.nan, data))


class PreprocessDetrend:

    def __init__(
            self,
            stop_mode: Optional[str] = None,
            metadata_response_group_by: str = None):
        self.stop_mode = stop_mode
        self.metadata_response_group_by = metadata_response_group_by

    @staticmethod
    def _detrend(arr, indicator_train):
        x = np.arange(len(arr))[indicator_train]
        to_fit = arr[indicator_train]
        to_fit = np.ma.masked_invalid(to_fit)
        p = np.ma.polyfit(x, np.reshape(to_fit.shape[0], -1), deg=1)
        #      (1, num_columns)            (num_rows, 1)
        lines = np.reshape(p[0], (1, -1)) * np.reshape(np.arange(len(arr)), (-1, 1)) + np.reshape(p[1], (1, -1))
        lines = np.reshape(lines, arr.shape)
        return arr - lines

    def __call__(self, loaded_data_tuple, metadata):

        indicator_train = _indicator_from_examples(len(loaded_data_tuple.data), loaded_data_tuple.train, self.stop_mode)

        if self.metadata_response_group_by is not None:
            if metadata is None or self.metadata_response_group_by not in metadata:
                raise ValueError('metadata_response_group_by {} not found in metadata'.format(
                    self.metadata_response_group_by))
            group_by = np.unique(metadata[self.metadata_response_group_by])
            groups = np.unique(group_by)
            data = np.copy(loaded_data_tuple.data)
            for group in groups:
                indicator_group = group == group_by
                group_data = data[indicator_group]
                group_indicator_train = indicator_train[indicator_group]
                group_data = PreprocessDetrend._detrend(group_data, group_indicator_train)
                data[indicator_group] = group_data
        else:
            data = PreprocessDetrend._detrend(loaded_data_tuple.data, indicator_train)

        return replace(loaded_data_tuple, data=data)


class PreprocessDiscretize:

    # noinspection PyShadowingBuiltins
    def __init__(self, bins=10, range=None, use_one_hot=True):
        self.bins = bins
        self.range = range
        self.use_one_hot = use_one_hot

    # noinspection PyShadowingBuiltins
    def __call__(self, loaded_data_tuple, metadata):
        bin_edges = np.histogram_bin_edges(loaded_data_tuple.data, self.bins, range)
        if np.isscalar(self.bins):
            bin_edges = bin_edges[1:]
        data = np.digitize(loaded_data_tuple.data, bin_edges, right=True)
        if self.use_one_hot:
            one_hot = np.zeros(data.shape + (len(bin_edges) + 1,), data.dtype)
            one_hot = np.reshape(one_hot, (-1, one_hot.shape[-1]))
            for idx, bin in enumerate(np.reshape(data, -1)):
                one_hot[idx, bin] = 1
            data = np.reshape(one_hot, data.shape + (one_hot.shape[-1],))
        return replace(loaded_data_tuple, data=data)


class PreprocessBaseline:

    def __init__(self, num_baseline):
        """
        Computes a running mean using a window of num_baseline values and subtracts this running mean
        from the data. This completely ignores example boundaries. Validation/test examples are removed if the baselines
        from those examples would overlap with train examples
        """
        self.num_baseline = num_baseline

    def _find_max_mins(self, examples, keep_if_greater_than=None):
        if examples is None:
            return None, None, examples
        data_ids = np.concatenate([ex.data_ids for ex in examples])
        data_ids = data_ids[data_ids >= 0]
        max_id = np.max(data_ids)
        min_id = np.min(data_ids)
        if keep_if_greater_than is None:
            return max_id, min_id, examples
        clean = list()
        for ex in examples:
            ex_min = np.min(ex.data_idx[ex.data_ids >= 0])
            if ex_min - self.num_baseline > keep_if_greater_than:
                clean.append(ex)
        return max_id, min_id, clean

    def _compute_baseline(self, arr):
        indicator_nan = np.isnan(arr)
        result = np.cumsum(np.where(indicator_nan, 0, arr), axis=0)
        if len(result) > self.num_baseline:
            result[self.num_baseline:] = result[self.num_baseline:] - result[:-self.num_baseline]
        counts = np.cumsum(np.logical_not(indicator_nan))
        if len(counts) > self.num_baseline:
            counts[self.num_baseline:] = counts[self.num_baseline:] - counts[:-self.num_baseline]
        return result / counts

    def _subtract_baseline(self, examples, data):
        if examples is None:
            return
        data_ids = np.concatenate([ex.data_ids for ex in examples])
        data_ids = data_ids[data_ids >= 0]
        baseline = self._compute_baseline(data[data_ids])
        data[data_ids] = data[data_ids] - baseline

    def __call__(self, loaded_data_tuple, metadata):
        max_train, _, _ = self._find_max_mins(loaded_data_tuple.train)
        _, _, clean_validation = self._find_max_mins(loaded_data_tuple.validation, keep_if_greater_than=max_train)
        loaded_data_tuple = replace(loaded_data_tuple, validation=clean_validation)
        _, _, clean_test = self._find_max_mins(loaded_data_tuple.test, keep_if_greater_than=max_train)
        loaded_data_tuple = replace(loaded_data_tuple, test=clean_test)

        data = np.copy(loaded_data_tuple.data)

        self._subtract_baseline(loaded_data_tuple.train, data)
        self._subtract_baseline(loaded_data_tuple.validation, data)
        self._subtract_baseline(loaded_data_tuple.test, data)

        return replace(loaded_data_tuple, data=data)


class PreprocessSequenceStandardize:

    def __init__(self, stop_mode):
        self.stop_mode = stop_mode

    def __call__(self, loaded_data_tuple, metadata):

        data = np.copy(loaded_data_tuple.data)
        for ex in chain(loaded_data_tuple.train, loaded_data_tuple.validation, loaded_data_tuple.test):

            data_indices = ex.data_ids
            compute_indices = data_indices
            if self.stop_mode == 'content':
                compute_indices = np.where(ex.is_stop, -1, compute_indices)
            elif self.stop_mode == 'stop':
                compute_indices = np.where(ex.is_stop, compute_indices, -1)
            elif self.stop_mode is not None:
                raise ValueError('Unable to understand stop_mode: {}'.format(self.stop_mode))

            data_indices = data_indices[data_indices >= 0]
            compute_indices = compute_indices[compute_indices >= 0]

            compute = data[compute_indices]
            mean = np.mean(compute, axis=0, keepdims=True)
            std = np.std(compute, axis=0, keepdims=True)

            data[data_indices] = (data[data_indices] - mean) / std

        return replace(loaded_data_tuple, data=data)


class PreprocessDiff:

    def __init__(self, fill_value=0):
        self.fill_value = fill_value

    def __call__(self, loaded_data_tuple, metadata):
        data = loaded_data_tuple.data
        padding = np.full((1,) + data.shape[1:], self.fill_value, data.dtype)
        return replace(loaded_data_tuple, data=np.concatenate([padding, np.diff(data, axis=0)]))


class PreprocessPCA:

    def __init__(
            self,
            feature_axis: int = 1,  # features with respect to PCA, e.g. subjects
            stop_mode: Optional[str] = None):
        self.feature_axis = feature_axis
        self.stop_mode = stop_mode

    def __call__(self, loaded_data_tuple, metadata):

        indicator_train = _indicator_from_examples(len(loaded_data_tuple.data), loaded_data_tuple.train, self.stop_mode)

        all_values = loaded_data_tuple.data
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
            train_values = current[indicator_train]
            pca.fit(train_values)
            result.append(pca.transform(current))

        # -> (task, samples, 1)
        result = np.array(result)
        # -> (samples, task, 1)
        result = np.transpose(result, (1, 0, 2))
        # -> (samples, ...)
        result = np.reshape(result, result_shape)

        return replace(loaded_data_tuple, data=result)


class PreprocessStandardize:

    def __init__(
            self,
            average_axis: Optional[int] = 1,
            stop_mode: Optional[str] = None,
            metadata_response_group_by: Optional[str] = None):
        self.stop_mode = stop_mode
        self.average_axis = average_axis
        self.metadata_response_group_by = metadata_response_group_by

    def _standardize(self, data, indicator_train):

        valid_train_values = data[indicator_train]

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

        indicator_train = _indicator_from_examples(len(loaded_data_tuple.data), loaded_data_tuple.train, self.stop_mode)

        if self.metadata_response_group_by is not None:
            if metadata is None or self.metadata_response_group_by not in metadata:
                raise ValueError('metadata_response_group_by not found: {}'.format(
                    self.metadata_response_group_by))

            data = np.copy(loaded_data_tuple.data)
            group_by = metadata[self.metadata_response_group_by]
            groups = np.unique(group_by)
            for group in groups:
                indicator_group = group_by == group
                group_data = loaded_data_tuple.data[indicator_group]
                group_indicator_train = indicator_train[indicator_group]
                group_data = self._standardize(group_data, group_indicator_train)
                data[indicator_group] = group_data
        else:
            data = self._standardize(loaded_data_tuple.data, indicator_train)

        return replace(loaded_data_tuple, data=data)


class PreprocessMakeBinary:

    def __init__(self, threshold, dtype=np.int32):
        self.threshold = threshold
        self.dtype = dtype

    def __call__(self, loaded_data_tuple, metadata):
        data = loaded_data_tuple.data
        indicator_nan = np.isnan(data)
        safe_compare = np.where(indicator_nan, self.threshold - 1, data)
        indicator = np.logical_and(np.logical_not(indicator_nan), safe_compare >= self.threshold)
        if self.dtype == np.bool_ or self.dtype == bool:
            data = indicator
        else:
            data = np.where(indicator, np.ones(data.shape, self.dtype), np.zeros(data.shape, self.dtype))

        return replace(loaded_data_tuple, data=data)


class PreprocessNanMean:

    def __init__(self, axis: int = 1):
        self.axis = axis

    def __call__(self, loaded_data_tuple, metadata):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            data = np.nanmean(loaded_data_tuple.data, self.axis)

        return replace(loaded_data_tuple, data=data)


class PreprocessClip:

    def __init__(
            self, minimum=None, maximum=None, value_beyond_min=None, value_beyond_max=None):
        self.value_beyond_min = value_beyond_min
        self.value_beyond_max = value_beyond_max
        self.min = minimum
        self.max = maximum

    def __call__(self, loaded_data_tuple, metadata):
        data = loaded_data_tuple.data
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            if self.min is not None:
                if self.value_beyond_min is not None:
                    data = np.where(data < self.min, self.value_beyond_min, data)
                else:
                    data = np.maximum(self.min, data)
            if self.max is not None:
                if self.value_beyond_max is not None:
                    data = np.where(data > self.max, self.value_beyond_max, data)
                else:
                    data = np.minimum(self.max, data)
        return replace(loaded_data_tuple, data=data)


class PreprocessNanGeometricMean:

    def __init__(self, axis=1):
        self.axis = axis

    def __call__(self, loaded_data_tuple, metadata):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            data = np.exp(np.nanmean(np.log(loaded_data_tuple.data), axis=self.axis))

        return replace(loaded_data_tuple, data=data)


class PreprocessGaussianBlur:

    def __init__(self, sigma=1, axis=1, order=0, mode='reflect', cval=0.0, truncate=4.0):
        """
        This is meant to blue over a non-example axis, e.g. spatially in fMRI
        """
        self.sigma, self.axis, self.order, self.mode, self.cval, self.truncate = \
            sigma, axis, order, mode, cval, truncate

    def __call__(self, loaded_data_tuple, metadata):
        data = gaussian_filter1d(
            loaded_data_tuple.data,
            sigma=self.sigma, axis=self.axis, order=self.order, mode=self.mode, truncate=self.truncate)

        return replace(loaded_data_tuple, data=data)


class PreprocessCompress:

    def __init__(self, metadata_condition_name, compress_axis=1):
        self.metadata_condition_name, self.compress_axis = metadata_condition_name, compress_axis

    def __call__(self, loaded_data_tuple, metadata):
        if metadata is None or self.metadata_condition_name not in metadata:
            raise ValueError('Unable to find metadata_condition_name: {}'.format(self.metadata_condition_name))
        condition = metadata[self.metadata_condition_name]
        data = np.compress(condition, loaded_data_tuple.data, axis=self.compress_axis)
        return replace(loaded_data_tuple, data=data)


class PreprocessMany:

    def __init__(self, *steps):
        self.steps = steps

    def __call__(self, loaded_data_tuple, metadata):
        for step in self.steps:
            loaded_data_tuple = step(loaded_data_tuple, metadata)

        return loaded_data_tuple


@dataclass(frozen=True)
class PreparedData:
    train: Optional[Sequence[InputFeatures]] = None
    validation: Optional[Sequence[InputFeatures]] = None
    test: Optional[Sequence[InputFeatures]] = None
    data: Optional[Mapping[str, KindData]] = None
    field_specs: Optional[Mapping[str, FieldSpec]] = None


@dataclass(frozen=True)
class PreparedDataView:
    train: Optional[Sequence[InputFeatures]] = None
    validation: Optional[Sequence[InputFeatures]] = None
    test: Optional[Sequence[InputFeatures]] = None
    data: Optional[np.array] = None


def _make_examples_view(
        examples: Optional[Sequence[InputFeatures]], response_key: str) -> Optional[Sequence[InputFeatures]]:
    return [replace(ex, data_ids=ex.data_ids[response_key]) for ex in examples] if examples is not None else None


def _make_prepared_data_view(prepared_data: PreparedData, response_key: str) -> PreparedDataView:
    return PreparedDataView(
        train=_make_examples_view(prepared_data.train, response_key),
        validation=_make_examples_view(prepared_data.validation, response_key),
        test=_make_examples_view(prepared_data.test, response_key),
        data=prepared_data.data[response_key].data)


def _reconcile_view_examples(
        prepared_data_examples: Sequence[InputFeatures],
        view_examples: Sequence[InputFeatures],
        response_key: str):
    if prepared_data_examples is None:
        return
    view_examples = dict((ex.unique_id, ex.data_ids) for ex in view_examples) if view_examples is not None else {}
    for ex in prepared_data_examples:
        ex.data_ids = dict(ex.data_ids)  # copy MappingProxyType into a new dict
        if ex.unique_id not in view_examples:
            ex.data_ids[response_key] = -1 * np.ones_like(ex.data_ids[response_key])
        else:
            ex.data_ids[response_key] = view_examples[ex.unique_id]
        ex.data_ids = MappingProxyType(ex.data_ids)


def _reconcile_view(prepared_data: PreparedData, view: PreparedData, response_key: str):
    _reconcile_view_examples(prepared_data.train, view.train, response_key)
    _reconcile_view_examples(prepared_data.validation, view.validation, response_key)
    _reconcile_view_examples(prepared_data.test, view.test, response_key)
    prepared_data.data[response_key] = replace(prepared_data.data[response_key], data=view.data)


class DataPreparer(object):

    def __init__(
            self,
            seed: int,
            preprocess_dict: Mapping[str, Callable[[PreparedData, Optional[Mapping[str, np.array]]], PreparedData]],
            split_function_dict: Mapping[
                str, Callable[
                    [RawData, np.random.RandomState],
                    Tuple[
                        Optional[Sequence[InputFeatures]],
                        Optional[Sequence[InputFeatures]],
                        Optional[Sequence[InputFeatures]]]]]):
        self._seed = seed
        self._random_state = dict()
        self._prepared_cache = dict()
        self._preprocess_dict = dict(preprocess_dict) if preprocess_dict is not None else None
        self._split_function_dict = dict(split_function_dict) if split_function_dict is not None else None

    def prepare(self, raw_data_dict: Mapping[str, RawData]) -> Mapping[str, PreparedData]:
        result = OrderedDict()
        metadata = OrderedDict()

        for k in raw_data_dict:
            metadata[k] = raw_data_dict[k].metadata
            if raw_data_dict[k].is_pre_split:
                result[k] = PreparedData(
                    raw_data_dict[k].input_examples,
                    raw_data_dict[k].validation_input_examples,
                    raw_data_dict[k].test_input_examples,
                    raw_data_dict[k].response_data,
                    field_specs=raw_data_dict[k].field_specs)
            elif (self._split_function_dict is not None
                    and k in self._split_function_dict and self._split_function_dict[k] is not None):
                if k not in self._random_state:
                    self._random_state[k] = np.random.RandomState(self._seed)
                train_input_examples, validation_input_examples, test_input_examples = self._split_function_dict[k](
                    raw_data_dict[k], self._random_state[k])
                result[k] = PreparedData(
                    train_input_examples, validation_input_examples, test_input_examples,
                    raw_data_dict[k].response_data,
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
                    raw_data_dict[k].response_data,
                    field_specs=raw_data_dict[k].field_specs)
                result[k] = loaded_data_tuple

        def _get_preprocessor(preprocess_dict, corpus_key, response_key, kind):
            if preprocess_dict is None:
                return None
            if response_key in preprocess_dict:
                return preprocess_dict[response_key]
            if kind in preprocess_dict:
                return preprocess_dict[kind]
            if corpus_key in preprocess_dict:
                return preprocess_dict[corpus_key]
            return None

        for k in result:
            needs_preprocess = not raw_data_dict[k].is_pre_split or k not in self._prepared_cache
            if needs_preprocess:
                for response_k in result[k].data:
                    preprocessor = _get_preprocessor(
                        self._preprocess_dict, k, response_k, result[k].data[response_k].kind)
                    if preprocessor is not None:
                        processed = preprocessor(_make_prepared_data_view(result[k], response_k), metadata[k])
                        _reconcile_view(result[k], processed, response_k)
                if raw_data_dict[k].is_pre_split:
                    self._prepared_cache[k] = result[k]
            elif raw_data_dict[k].is_pre_split:
                result[k] = self._prepared_cache[k]

        return result
