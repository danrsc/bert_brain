import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
from dataclasses import replace
from typing import Optional
import numpy as np
from scipy.stats import boxcox
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import sosfilt
from sklearn.decomposition import PCA

__all__ = [
    'PreprocessBoxcox',
    'PreprocessLog',
    'PreprocessDetrend',
    'PreprocessDiscretize',
    'PreprocessBaseline',
    'PreprocessFeatureStandardize',
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
    'PreprocessSoSFilter',
    'PreprocessMany']


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


def _unsorted_group_by(items, group_by_fn):
    groups = dict()
    for item in items:
        key = group_by_fn(item)
        if key not in groups:
            groups[key] = list()
        groups[key].append(item)
    for key in sorted(groups):
        yield key, groups[key]


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
            metadata_example_group_by: str = None,
            train_on_all: bool = False):
        self.stop_mode = stop_mode
        self.metadata_example_group_by = metadata_example_group_by
        self.train_on_all = train_on_all

    @staticmethod
    def _detrend(arr, indicator_train):
        x = np.arange(len(arr))
        to_fit = arr

        if indicator_train is not None:
            x = x[indicator_train]
            to_fit = to_fit[indicator_train]

        if len(to_fit) == 0:
            return arr

        to_fit = np.reshape(np.ma.masked_invalid(to_fit), (to_fit.shape[0], -1))

        # some columns might not have any data
        indicator_can_fit = np.logical_not(np.all(to_fit.mask, axis=0))
        to_fit = to_fit[:, indicator_can_fit]

        p = np.ma.polyfit(x, to_fit, deg=1)

        filled_p = np.zeros((p.shape[0], len(indicator_can_fit)), p.dtype)
        filled_p[:, indicator_can_fit] = p
        p = filled_p

        #      (1, num_columns)            (num_rows, 1)
        lines = np.reshape(p[0], (1, -1)) * np.reshape(np.arange(len(arr)), (-1, 1)) + np.reshape(p[1], (1, -1))
        lines = np.reshape(lines, arr.shape)
        return arr - lines

    def __call__(self, loaded_data_tuple, metadata):

        train_examples = loaded_data_tuple.train
        if self.train_on_all:
            train_examples = chain(loaded_data_tuple.train, loaded_data_tuple.validation, loaded_data_tuple.test)
        indicator_train = _indicator_from_examples(
            len(loaded_data_tuple.data), train_examples, self.stop_mode)

        if self.metadata_example_group_by is not None:
            if metadata is None or self.metadata_example_group_by not in metadata:
                raise ValueError('metadata_example_group_by {} not found in metadata'.format(
                    self.metadata_example_group_by))
            data = np.copy(loaded_data_tuple.data)
            grouped_examples = _unsorted_group_by(
                chain(loaded_data_tuple.train, loaded_data_tuple.validation, loaded_data_tuple.test),
                lambda ex: metadata[self.metadata_example_group_by][ex.unique_id])
            for group, group_examples in grouped_examples:
                indicator_group = _indicator_from_examples(len(data), group_examples)
                group_data = data[indicator_group]
                group_indicator_train = indicator_train[indicator_group] if indicator_train is not None else None
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


class PreprocessFeatureStandardize:

    def __init__(self):
        pass

    def __call__(self, loaded_data_tuple, metadata):
        d = np.reshape(loaded_data_tuple.data, (loaded_data_tuple.data.shape[0], -1))
        d = (d - np.nanmean(d, axis=1, keepdims=True)) / np.nanstd(d, axis=1, keepdims=True)
        return replace(loaded_data_tuple, data=np.reshape(d, loaded_data_tuple.data.shape))


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
            (all_values.shape[0], int(np.prod(all_values.shape[1:-1])), all_values.shape[-1]))
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
            metadata_example_group_by: Optional[str] = None,
            train_on_all: Optional[bool] = False):
        self.stop_mode = stop_mode
        self.average_axis = average_axis
        self.metadata_example_group_by = metadata_example_group_by
        self.train_on_all = train_on_all

    def _standardize(self, data, indicator_train):

        valid_train_values = data
        if indicator_train is not None:
            valid_train_values = data[indicator_train]

        if len(valid_train_values) == 0:
            raise ValueError('No training values')

        with warnings.catch_warnings():
            # catch 'Mean of emtpy slice' warning here; for example if a participant has no data
            # within the training examples
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            pre_average_mean = np.nanmean(valid_train_values, axis=0, keepdims=True)
            pre_average_std = np.nanstd(valid_train_values, axis=0, keepdims=True)

        transformed_data = np.divide(data - pre_average_mean, pre_average_std, where=pre_average_std != 0)

        if self.average_axis is not None:
            standardized_train_values = transformed_data
            if indicator_train is not None:
                standardized_train_values = standardized_train_values[indicator_train]
            with warnings.catch_warnings():
                # catch 'Mean of emtpy slice' warning here
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                standardized_train_values = np.nanmean(standardized_train_values, axis=self.average_axis)
                transformed_data = np.nanmean(transformed_data, axis=self.average_axis)
            post_average_mean = np.nanmean(standardized_train_values, axis=0, keepdims=True)
            post_average_std = np.nanstd(standardized_train_values, axis=0, keepdims=True)
            transformed_data = np.divide(
                transformed_data - post_average_mean, post_average_std, where=post_average_std != 0)

        return transformed_data

    def __call__(self, loaded_data_tuple, metadata):

        train_examples = loaded_data_tuple.train
        if self.train_on_all:
            train_examples = chain(loaded_data_tuple.train, loaded_data_tuple.validation, loaded_data_tuple.test)
        indicator_train = _indicator_from_examples(
            len(loaded_data_tuple.data), train_examples, self.stop_mode)

        if self.metadata_example_group_by is not None:
            if metadata is None or self.metadata_example_group_by not in metadata:
                raise ValueError('metadata_example_group_by not found: {}'.format(self.metadata_example_group_by))
            if self.average_axis is None:   # we're going to keep the shape
                data = np.full(loaded_data_tuple.data.shape, np.nan)
            else:
                data = np.full(
                    loaded_data_tuple.data.shape[:self.average_axis]
                    + loaded_data_tuple.data.shape[(self.average_axis + 1):], np.nan)
            grouped_examples = _unsorted_group_by(
                chain(loaded_data_tuple.train, loaded_data_tuple.validation, loaded_data_tuple.test),
                lambda ex: metadata[self.metadata_example_group_by][ex.unique_id])
            for group, group_examples in grouped_examples:
                indicator_group = _indicator_from_examples(len(data), group_examples)
                group_data = loaded_data_tuple.data[indicator_group]
                if len(group_data) == 0:
                    continue
                group_indicator_train = indicator_train[indicator_group] if indicator_train is not None else None
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


class PreprocessSoSFilter:

    def __init__(self, sos, axis=0):
        """
        Apply scipy.signal.sosfilt to data
        Args:
            sos: iirfilter created with output='sos',
                e.g.
                    A high-pass butterworth filter for a sampling rate of 0.5 Hz
                    and cutoff 0.2 Hz
                signal.butter(10, 0.2, 'hp', fs=0.5, output='sos')
            axis: Which axis to apply along
        """
        self.sos = sos
        self.axis = axis

    def __call__(self, loaded_data_tuple, metadata):
        return replace(loaded_data_tuple, data=sosfilt(self.sos, loaded_data_tuple.data, axis=self.axis))


class PreprocessMany:

    def __init__(self, *steps):
        self.steps = steps

    def __call__(self, loaded_data_tuple, metadata):
        for step in self.steps:
            loaded_data_tuple = step(loaded_data_tuple, metadata)

        return loaded_data_tuple
