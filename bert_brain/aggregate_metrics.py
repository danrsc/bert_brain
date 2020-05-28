import inspect
import os
import warnings
from collections import OrderedDict
import dataclasses
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from tqdm.auto import tqdm

import numpy as np
from scipy.special import logsumexp
from scipy.stats.mstats import rankdata
from scipy.stats import ttest_rel, ttest_1samp

from .experiments import singleton_variation, task_hash
from .modeling import critic_types, NamedTargetMaskedLossBase
from .result_output import read_predictions, get_output_keys
from .data_sets import DataIdDataset
from .meta_data import TextLabels

__all__ = [
    'Aggregator',
    'read_variation_results',
    'load_switched_corpus',
    'nan_pearson',
    'regression_handler',
    'class_handler',
    'bincount_axis',
    'nan_rank_accuracy',
    'make_prediction_handler',
    'get_field_predictions',
    'assemble_indexed_predictions',
    'k_vs_k',
    'compute_statistics_on_aggregated',
    'StatisticsSpec',
    'StatisticsResult',
    'spatial_neighbor_edges']


class Aggregator:
    def __init__(self):
        """
        Helper class to aggregate metrics over runs etc.
        """
        self._field_values = None
        self._counts = None

    def update(self, result, is_sequence):
        if self._field_values is None:
            self._field_values = OrderedDict()
            self._counts = OrderedDict()
            if dataclasses.is_dataclass(result):
                for field in dataclasses.fields(result):
                    self._field_values[field.name] = list()
                    self._counts[field.name] = list()
            else:
                for field in result:
                    self._field_values[field] = list()
                    self._counts[field] = list()

        if dataclasses.is_dataclass(result):
            result = dataclasses.asdict(result)
        for field in result:
            if field not in self._field_values:
                raise ValueError('Unexpected field in result: {}'.format(field))
            if result[field] is None:
                self._counts[field].append(0)
            elif np.ndim(result[field]) == 0:
                self._field_values[field].append(result[field])
                self._counts[field].append(1)
            elif is_sequence:
                self._field_values[field].extend(result[field])
                self._counts[field].append(len(result[field]))
            else:
                self._field_values[field].append(result[field])
                self._counts[field].append(1)

    def __contains__(self, item):
        return item in self._field_values

    def __iter__(self):
        for k in self._field_values:
            yield k

    def __getitem__(self, item):
        return self._field_values[item]

    def value_dict(self, names=None, fn=None, value_on_key_error=None):
        if names is None:
            if fn is None:
                return OrderedDict(self._field_values)
            return OrderedDict((k, fn(self._field_values[k])) for k in self._field_values)
        if isinstance(names, str):
            names = [names]
        result = OrderedDict()
        for name in names:
            if value_on_key_error is not None and name not in self._field_values:
                result[name] = value_on_key_error
            else:
                result[name] = fn(self._field_values[name]) if fn is not None else self._field_values[name]
        return result

    def values(self, name, fn=None):
        if fn is None:
            return self._field_values[name]
        return fn(self._field_values[name])

    def counts(self, name):
        return self._counts[name]


def read_no_cluster_data(path):
    with np.load(path) as loaded:
        unique_ids = loaded['unique_ids']
        lengths = loaded['lengths']
        data_ids = loaded['data_ids']
        splits = np.cumsum(lengths)[:-1]
        data_ids = np.split(data_ids, splits)
        return unique_ids, data_ids, loaded['data']


def expand_predictions(prediction, cluster_ids):
    is_prediction_1d = len(prediction.shape) == 1
    if is_prediction_1d:
        prediction = np.expand_dims(prediction, 0)
    expanded = np.zeros((prediction.shape[0], np.prod(cluster_ids.shape)), prediction.dtype)
    for idx, c in enumerate(np.unique(cluster_ids)):
        indicator = cluster_ids == c
        expanded[:, indicator] = prediction[:, idx]
    if is_prediction_1d:
        return np.reshape(expanded, cluster_ids.shape)
    else:
        return np.reshape(expanded, (prediction.shape[0],) + cluster_ids.shape)


_process_paths_obj = None
_process_text_labels = dict()


def _set_process_paths_obj(paths_obj):
    global _process_paths_obj
    _process_paths_obj = paths_obj


def _read_variation_parallel_helper(item):
    (variation_name, index_run,
     compute_scalar, k_vs_k_feature_axes, loss_handler_kwargs) = item
    paths_obj = _process_paths_obj
    (variation_set_name, training_variation_name), settings = singleton_variation(variation_name)
    output_dir = os.path.join(paths_obj.result_path, variation_set_name, task_hash(settings))
    model_dir = os.path.join(paths_obj.model_path, variation_set_name, task_hash(settings), 'run_{}'.format(index_run))
    if not os.path.exists(os.path.join(output_dir, 'run_{}'.format(index_run), 'completed.txt')):
        return index_run, None
    validation_dir = os.path.join(output_dir, 'run_{}'.format(index_run), 'validation_predictions')
    output_results_by_name = read_predictions(validation_dir)
    run_results = dict()
    for name in output_results_by_name:
        no_cluster_path = os.path.join(model_dir, '{}_no_cluster_to_disk.npz'.format(name))
        cluster_id_path = os.path.join(model_dir, 'kmeans_clusters_{}.npy'.format(name))
        cluster_ids = None
        no_cluster_unique_ids = None
        no_cluster_data_ids = None
        no_cluster_data = None
        if os.path.exists(cluster_id_path) and os.path.exists(no_cluster_path):
            cluster_ids = np.load(cluster_id_path)
            no_cluster_unique_ids, no_cluster_data_ids, no_cluster_data = read_no_cluster_data(no_cluster_path)
        output_results = output_results_by_name[name]
        run_aggregated = Aggregator()
        loss = None
        is_active_loss = True
        for output_result in output_results:
            if not output_result.is_active_loss:
                is_active_loss = False
                break
            if loss is None:
                loss = output_result.critic_type
            else:
                assert (loss == output_result.critic_type)
            if cluster_ids is not None:
                output_result.prediction = expand_predictions(output_result.prediction, cluster_ids)
                output_result.mask = expand_predictions(output_result.mask, cluster_ids)
                index_unique_id = np.where(output_result.unique_id == no_cluster_unique_ids)[0]
                assert(len(index_unique_id) == 1)
                index_unique_id = index_unique_id[0]
                data_ids = no_cluster_data_ids[index_unique_id]
                data_ids = data_ids[data_ids >= 0]
                seen = set()
                unique_data_ids = list()
                for d in data_ids:
                    if d not in seen:
                        unique_data_ids.append(d)
                        seen.add(d)
                assert(len(unique_data_ids) == output_result.target.shape[0])
                output_result.target = np.array(list([no_cluster_data[d] for d in unique_data_ids]))
            run_aggregated.update(output_result, is_sequence=output_result.sequence_type != 'single')

        if not is_active_loss:
            continue

        loss_handler_kwargs = dict(loss_handler_kwargs)
        if isinstance(k_vs_k_feature_axes, dict):
            if name in k_vs_k_feature_axes:
                loss_handler_kwargs['k_vs_k_feature_axes'] = k_vs_k_feature_axes[name]
            else:
                loss_handler_kwargs['k_vs_k_feature_axes'] = -1
        else:
            loss_handler_kwargs['k_vs_k_feature_axes'] = k_vs_k_feature_axes
        handler = make_prediction_handler(loss, loss_handler_kwargs)
        handler_result = handler(run_aggregated)
        is_many = isinstance(handler_result, list)
        if not is_many:
            handler_result = [handler_result]
            if compute_scalar:
                for index_result_dict in range(len(handler_result)):
                    result_dict = handler_result[index_result_dict]
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=RuntimeWarning)
                        handler_result[index_result_dict] = dict((k, np.nanmean(result_dict[k])) for k in result_dict)
        if not is_many:
            run_results[name] = handler_result[0]
        else:
            if variation_name not in _process_text_labels:
                _process_text_labels[variation_name] = TextLabels(paths_obj, variation_name, index_run)
            label_maker = _process_text_labels[variation_name]
            text_labels = label_maker.labels(name, len(handler_result))
            for text_label, result_dict in zip(text_labels, handler_result):
                run_results[text_label] = result_dict
    return index_run, run_results


def load_switched_corpus(variation_name, replacement_corpus, index_run):
    _, settings = singleton_variation(variation_name)
    is_found = False
    for settings_corpus in settings.corpora:
        if settings_corpus.corpus_key == replacement_corpus.corpus_key:
            is_found = True
            break
    if not is_found:
        raise ValueError('corpus {} not in settings for {}'.format(replacement_corpus.corpus_key, variation_name))

    from run_variations import _io_setup
    corpus_dataset_factory, paths = _io_setup(variation_name, settings)

    data_set_path = corpus_dataset_factory.maybe_make_data_set_files(
        index_run,
        replacement_corpus,
        settings.preprocessors,
        settings.get_split_function(replacement_corpus, index_run),
        settings.preprocess_fork_fn,
        False,
        paths,
        settings.max_sequence_length,
        settings.create_meta_train_dataset)

    train_data, validation_data, test_data, meta_train_data = (
        DataIdDataset(
            data_set_path,
            which,
            DataIdDataset.get_init_metadata(data_set_path).max_sequence_length,
            settings.all_loss_tasks,
            data_id_in_batch_keys=settings.data_id_in_batch_keys,
            filter_when_not_in_loss_keys=settings.filter_when_not_in_loss_keys,
            field_spec_replacers=settings.field_spec_replacers)
        for which in ('train', 'validation', 'test', 'meta_train'))

    return train_data, validation_data, test_data, meta_train_data


def read_variation_results(
        paths, variation_name, index_run=None, compute_scalar=True, k_vs_k_feature_axes=-1, **loss_handler_kwargs):

    _, settings = singleton_variation(variation_name)
    if index_run is None:
        runs = range(settings.num_runs)
    elif callable(index_run):
        runs = index_run(settings.num_runs)
    elif np.ndim(index_run) == 0:
        runs = [index_run]
    else:
        runs = index_run

    task_arguments = [(variation_name, i, compute_scalar, k_vs_k_feature_axes, loss_handler_kwargs) for i in runs]

    with ThreadPoolExecutor(initializer=_set_process_paths_obj, initargs=(paths,)) as ex:
        mapped = ex.map(_read_variation_parallel_helper, task_arguments)
    # mapped = map(_read_variation_parallel_helper, task_arguments)

    has_warned = False
    count_runs = 0
    aggregated = dict()
    for index_run, run_results in tqdm(mapped):
        if run_results is None:
            if not has_warned:
                print('Warning: results incomplete. Some output files not found')
            has_warned = True
            continue

        count_runs += 1
        for name in run_results:
            if name not in aggregated:
                aggregated[name] = Aggregator()
            aggregated[name].update(run_results[name], is_sequence=False)

    return aggregated, count_runs, settings


def nan_pearson(x, y, axis=0, keepdims=False):
    if not np.array_equal(x.shape, y.shape):
        raise ValueError('x and y must be the same shape')
    if np.isscalar(x):
        raise ValueError('x and y must not be scalar')
    if np.prod(x.shape) == 0:
        result = np.full_like(x, np.nan)
        if x.shape[axis] < 1:
            print(x.shape)
            raise ValueError('x and y must have at least 2 values')
        result = np.take(result, [0], axis=axis)
        if not keepdims:
            result = np.squeeze(result, axis=axis)
        return result
    with warnings.catch_warnings():
        # suppress ddof < 1 for slice
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        x = x - np.nanmean(x, axis=axis, keepdims=True)
        y = y - np.nanmean(y, axis=axis, keepdims=True)
        std_x = np.nanstd(x, axis=axis, keepdims=True, ddof=1)
        std_y = np.nanstd(y, axis=axis, keepdims=True, ddof=1)
    total = np.nansum(
        np.divide(x, std_x, where=std_x != 0) * np.divide(y, std_y, where=std_y != 0),
        axis=axis, keepdims=True)
    counts = np.sum(np.logical_and(np.logical_not(np.isnan(x)), np.logical_not(np.isnan(y))), axis=axis, keepdims=True)
    result = np.divide(total, (counts - 1), where=counts > 1)
    result = np.where(np.logical_and(np.logical_and(std_x != 0, std_y != 0), counts > 1),
                      result,
                      np.full_like(result, np.nan))
    if not keepdims:
        result = np.squeeze(result, axis)
    return result


def nan_rank_accuracy(scores, target):
    ranks = np.where(
        np.isnan(target),
        np.nan,
        np.squeeze(
            np.take_along_axis(
                rankdata(-scores, axis=-1),
                np.expand_dims(np.where(np.isnan(target), 0, target).astype(np.intp), -1), axis=-1),
            axis=-1))
    return 1 - (ranks - 1) / (scores.shape[-1] - 1)


def aggregator_regression_handler(aggregator, k_vs_k_num_samples=0, k_vs_k_k=20, k_vs_k_feature_axes=-1):
    target = np.array(aggregator.values('target'))
    predictions = np.array(aggregator.values('prediction'))
    mask = np.array(aggregator.values('mask'))

    target_counts = np.array(aggregator.counts('target'))
    prediction_counts = np.array(aggregator.counts('prediction'))
    assert(np.array_equal(target_counts, prediction_counts))

    splits = None
    if np.any(target_counts > 1):
        splits = np.cumsum(target_counts)[:-1]

    return regression_handler(
        predictions, target, mask, k_vs_k_num_samples, k_vs_k_k, k_vs_k_feature_axes, splits, is_single_example=False)


def _corr(x, y):
    indicator_valid = np.logical_and(np.isfinite(x), np.isfinite(y))
    x = np.where(indicator_valid, x, np.nan)
    y = np.where(indicator_valid, y, np.nan)
    std_x = np.nanstd(x, axis=0, keepdims=True, ddof=1)
    std_y = np.nanstd(y, axis=0, keepdims=True, ddof=1)
    x = np.divide(x - np.nanmean(x, axis=0, keepdims=True), std_x, where=std_x > 0)
    y = np.divide(y - np.nanmean(y, axis=0, keepdims=True), std_y, where=std_y > 0)
    return np.nanmean(x * y, axis=0)


def regression_handler(
        predictions, target, mask,
        k_vs_k_num_samples=0, k_vs_k_k=20, k_vs_k_feature_axes=-1,
        splits=None, is_single_example=False, class_wise=False):
    # class_wise is not used in regression_handler, but we keep the signature consistent with class_handler

    if is_single_example and len(target) > 1:
        seq_r = nan_pearson(predictions, target)
    elif splits is not None:
        seq_r = list()
        for seq_predictions, seq_target in zip(np.split(predictions, splits), np.split(target, splits)):
            seq_r.append(nan_pearson(seq_predictions, seq_target))
        seq_r = np.array(seq_r)
        with warnings.catch_warnings():
            # filter mean of empty slice
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            seq_r = np.nanmean(seq_r, axis=0)
    else:
        seq_r = np.nan

    if len(mask) > 0:
        assert(len(mask) == len(target))
        masked_target = np.where(mask, target, np.nan)
    else:
        masked_target = target

    variance = np.nanvar(masked_target, axis=0)

    mu = np.nanmean(masked_target, axis=0)
    mean_abs_deviation = np.nanmean(np.abs(masked_target - mu))

    mse = np.nanmean(np.square(predictions - masked_target), axis=0)
    mae = np.nanmean(np.abs(predictions - masked_target), axis=0)

    variance = np.where(variance < 1e-8, np.nan, variance)
    mean_abs_deviation = np.where(mean_abs_deviation < 1e-8, np.nan, mean_abs_deviation)

    result = dict(
        mse=mse,
        mae=mae,
        pove=1 - (mse / variance),
        povu=(mse / variance),
        r=_corr(predictions, masked_target),
        pode=1 - (mse / mean_abs_deviation),
        podu=(mae / mean_abs_deviation),
        variance=variance,
        mad=mean_abs_deviation,
        r_seq=seq_r)

    if k_vs_k_num_samples > 0:
        k_vs_k_mask = np.reshape(mask, (mask.shape[0], -1))
        # TODO: hack because the mask is sometimes wrong
        k_vs_k_mask = np.full_like(k_vs_k_mask, True)
        if not np.all(k_vs_k_mask == k_vs_k_mask[:, 0:1]):
            raise ValueError('For k_vs_k, the mask must be the same for all features')
        k_vs_k_mask = k_vs_k_mask[:, 0]
        accuracy = k_vs_k(
            predictions[k_vs_k_mask], target[k_vs_k_mask], k=k_vs_k_k, num_samples=k_vs_k_num_samples,
            feature_axes=k_vs_k_feature_axes)
        result['{0}_vs_{0}'.format(k_vs_k_k)] = np.mean(accuracy, axis=0)

    return result


def bincount_axis(x, weights=None, minlength=None, axis=-1):
    """
    Similar to np.bincount, but applied along an axis. By using weights, this function can do sums along contiguous
    segments of an array with variable numbers of elements (in which case x is essentially the label for a segment
    we are summing over). Without weights, this can be used to count the number of elements within a segment.
    See the documentation for np.bincount
    Args:
        x: Input array
        weights: Weights array, same shape as x.
        minlength: A minimum number of bins for the output array, defaults to np.max(x) + 1
        axis: Which axis to apply the bincount over

    Returns:
        out: The result of binning the input array
    """

    if minlength is None:
        minlength = np.max(x) + 1

    if axis < 0:
        axis += len(x.shape)
    transpose_axes = list(range(len(x.shape)))
    transpose_axes = transpose_axes[:axis] + transpose_axes[axis+1:] + [axis]
    x = np.transpose(x, transpose_axes)
    shape = x.shape
    x = np.reshape(x, (-1, x.shape[-1]))
    x += np.expand_dims(minlength * np.arange(x.shape[0]), 1)
    num_bins = minlength * x.shape[0]
    x = np.reshape(x, (-1,))
    if weights is not None:
        weights = np.transpose(weights, transpose_axes)
        weights = np.reshape(weights, (-1,))

    if weights is not None and np.iscomplexobj(weights):
        x_real = np.bincount(x, np.real(weights), num_bins)
        x_imag = np.bincount(x, np.imag(weights), num_bins)
        x = x_real + 1j * x_imag
    else:
        x = np.bincount(x, weights, num_bins)
    x = np.reshape(x, shape[:-1] + (minlength,))
    transpose_axes = list(range(len(x.shape)))
    transpose_axes = transpose_axes[:axis] + transpose_axes[-1:] + transpose_axes[axis:-1]
    return np.transpose(x, transpose_axes)


def aggregator_class_handler(aggregator, pos_weight=None, is_binary=False, class_wise=False):

    target = np.array(aggregator.values('target'))
    predictions = np.array(aggregator.values('prediction'))
    mask = np.array(aggregator.values('mask'))

    return class_handler(predictions, target, mask, pos_weight, is_binary, class_wise=class_wise)


def class_handler(
        predictions, target, mask, pos_weight=None, is_binary=False, is_single_example=False, class_wise=False):
    # is_single_example is not currently used; it is there so the caller can pass it without knowing
    # what the loss handler is

    if len(mask) != 0:
        assert(len(mask) == len(target))
        target = np.where(mask, target, np.nan)

    if is_binary:
        max_val = np.maximum(-predictions, 0)
        log_weight = 1
        if pos_weight is not None:
            log_weight = (pos_weight - 1) * target + 1
        cross_entropy = \
            predictions - predictions * target + max_val \
            + np.log(log_weight * (np.exp(-max_val) + np.exp(-predictions - max_val)))

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            cross_entropy = np.nanmean(cross_entropy, axis=0)

        indicator_valid = np.logical_not(np.isnan(target))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            target_positive = np.logical_and(np.greater(target, 0), indicator_valid)
            target_negative = np.logical_and(np.equal(target, 0), indicator_valid)
            predictions_positive = np.logical_and(np.greater_equal(predictions, 0), indicator_valid)
            predictions_negative = np.logical_and(np.less(predictions, 0), indicator_valid)

        true_positive = np.logical_and(predictions_positive, target_positive)
        true_negative = np.logical_and(predictions_negative, target_negative)
        false_positive = np.logical_and(predictions_positive, target_negative)
        false_negative = np.logical_and(predictions_negative, target_positive)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            precision = np.sum(true_positive, axis=0) / (np.sum(true_positive, axis=0) + np.sum(false_positive, axis=0))
        # nothing was classified as positive, define this to be precision 0.
        # where does something weird to scalar values...so we handle it separately
        if np.isscalar(precision):
            if np.isnan(precision):
                precision = np.array([0.])[0]
        else:
            precision = np.where(np.sum(true_positive, axis=0) + np.sum(false_positive, axis=0) == 0, 0., precision)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            recall = np.sum(true_positive, axis=0) / (np.sum(true_positive, axis=0) + np.sum(false_negative, axis=0))
        # either there are no real positive examples (define this to be recall 1),
        # or the predictions are nan (define this to be recall 0).
        if np.isscalar(recall):
            if np.isnan(recall):
                if np.sum(predictions_positive, axis=0) + np.sum(predictions_negative, axis=0) == 0:
                    recall = np.array([0.])[0]
                else:
                    recall = np.array([1.])[0]
        else:
            recall = np.where(np.sum(true_positive, axis=0) + np.sum(false_negative, axis=0) == 0, 1., recall)
            nan_prediction = np.sum(predictions_positive, axis=0) + np.sum(predictions_negative, axis=0) == 0
            recall = np.where(nan_prediction, 0., recall)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            valid_counts = np.sum(indicator_valid, axis=0)
            accuracy = np.divide(
                np.sum(true_positive, axis=0) + np.sum(true_negative, axis=0), valid_counts, where=valid_counts > 0)
            pos_acc = np.divide(np.sum(target_positive, axis=0), valid_counts, where=valid_counts > 0)
            neg_acc = np.divide(np.sum(target_negative, axis=0), valid_counts, where=valid_counts > 0)
        positive_better = np.sum(np.greater_equal(pos_acc, neg_acc)) > np.sum(np.less(pos_acc, neg_acc))
        mode_accuracy = pos_acc if positive_better else neg_acc
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            f1 = 2 * precision * recall / (precision + recall)
        if np.isscalar(f1):
            if precision + recall == 0:
                f1 = 0
        else:
            f1 = np.where(precision + recall == 0, 0, f1)

        poma = np.divide(accuracy, mode_accuracy, where=mode_accuracy != 0)

        return dict(
            xent=cross_entropy,
            acc=accuracy,
            macc=mode_accuracy,
            poma=poma,
            prec=precision,
            rec=recall,
            f1=f1)
    else:
        is_hard_label = len(predictions.shape) > len(target.shape) or target.shape[-1] == 1
        log_sum = logsumexp(predictions, axis=-1)
        # noinspection PyUnresolvedReferences
        log_sum = np.reshape(log_sum, log_sum.shape + (1,))
        log_softmax = predictions - log_sum
        predicted_class = np.argmax(predictions, axis=-1)
        if is_hard_label:
            bin_counts = bincount_axis(
                np.where(np.isnan(target), np.nanmax(target) + 1, target).astype(np.intp), axis=0)
            if len(bin_counts) == np.nanmax(target + 1):
                bin_counts = bin_counts[:-1]  # trim off the nan bin
            modes = np.argmax(bin_counts, axis=0)

            if class_wise:
                class_wise_results = list()
                for target_class in range(log_softmax.shape[-1]):
                    current_target = np.where(target == target_class, target, np.nan)
                    cross_entropy = np.where(
                        np.isnan(current_target),
                        np.nan,
                        np.squeeze(
                            np.take_along_axis(
                                -log_softmax,
                                np.expand_dims(
                                    np.where(np.isnan(current_target), 0, current_target).astype(np.intp), -1),
                                axis=-1),
                            axis=-1))
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=RuntimeWarning)
                        cross_entropy = np.nanmean(cross_entropy, axis=0)

                    denom_model = np.sum(
                        np.greater_equal(np.where(np.isnan(current_target), -1, current_target), 0), axis=0)

                    accuracy = np.where(denom_model == 0, 0, np.divide(
                        np.sum(np.equal(predicted_class, current_target), axis=0), denom_model, where=denom_model != 0))

                    denom_mode = np.sum(
                        np.greater_equal(np.where(np.isnan(current_target), -1, current_target), 0), axis=0)

                    mode_accuracy = np.where(denom_mode == 0, 0, np.divide(
                        np.sum(np.equal(modes, current_target), axis=0), denom_mode, where=denom_mode != 0))

                    poma = 1 if mode_accuracy == 0 else accuracy / mode_accuracy
                    rank_accuracy = np.nanmean(nan_rank_accuracy(predictions, target), axis=0)
                    class_wise_results.append(
                        dict(xent=cross_entropy, acc=accuracy, macc=mode_accuracy, poma=poma, racc=rank_accuracy))
                return class_wise_results
            else:
                cross_entropy = np.where(
                    np.isnan(target),
                    np.nan,
                    np.squeeze(
                        np.take_along_axis(
                            -log_softmax,
                            np.expand_dims(np.where(np.isnan(target), 0, target).astype(np.intp), -1), axis=-1),
                        axis=-1))
                cross_entropy = np.nanmean(cross_entropy, axis=0)
                accuracy = (np.sum(np.equal(predicted_class, target), axis=0)
                            / np.sum(np.greater_equal(np.where(np.isnan(target), -1, target), 0), axis=0))
                mode_accuracy = (np.sum(np.equal(modes, target), axis=0)
                                 / np.sum(np.greater_equal(np.where(np.isnan(target), -1, target), 0), axis=0))
                poma = accuracy / mode_accuracy
                rank_accuracy = np.nanmean(nan_rank_accuracy(predictions, target), axis=0)
                return dict(xent=cross_entropy, acc=accuracy, macc=mode_accuracy, poma=poma, racc=rank_accuracy)
        else:
            # soft class labels
            cross_entropy = np.nanmean(np.sum(-log_softmax * target, axis=-1), axis=0)
            max_values = np.max(target, axis=-1, keepdims=True)
            indicator_max = np.isclose(target, max_values)
            count_max = np.sum(indicator_max, axis=-1, keepdims=True)

            partial_credit = np.where(
                indicator_max,
                np.divide(1., count_max, where=count_max > 0),
                np.zeros(indicator_max.shape, target.dtype))

            partial_credit = np.where(count_max == 0, np.nan, partial_credit)
            constant_accuracy = np.nanmean(partial_credit, axis=0)
            mode_accuracy = np.max(constant_accuracy, axis=-1)

            partial_credit = np.reshape(partial_credit, (-1, partial_credit.shape[-1]))
            predicted_class = np.reshape(predicted_class, -1)
            partial_credit = np.array([c[p] for c, p in zip(partial_credit, predicted_class)])
            partial_credit = np.reshape(partial_credit, target.shape[:-1])
            accuracy = np.nanmean(partial_credit, axis=0)

            poma = accuracy / mode_accuracy
            rank_accuracy = np.nanmean(nan_rank_accuracy(predictions, target), axis=0)

            return dict(xent=cross_entropy, acc=accuracy, macc=mode_accuracy, poma=poma, racc=rank_accuracy)


def make_prediction_handler(which_loss, loss_kwargs=None, using_aggregator=True):
    if not hasattr(critic_types, which_loss):
        raise ValueError('Unknown value for which_loss. Known values are: {}'.format(critic_types.__all__))
    critic_type: NamedTargetMaskedLossBase = getattr(critic_types, which_loss)
    if critic_type.is_classification_loss():
        binary_classifier_types = frozenset(t.__name__ for t in [
            critic_types.NamedTargetStopWordAwareBinaryCrossEntropyWithLogits,
            critic_types.NamedTargetSingleBinaryCrossEntropyWithLogits])
        if critic_type.__name__ in binary_classifier_types:
            if using_aggregator:
                factory, factory_kwargs = aggregator_class_handler, dict(is_binary=True)
            else:
                factory, factory_kwargs = class_handler, dict(is_binary=True)
        else:
            if using_aggregator:
                factory, factory_kwargs = aggregator_class_handler, None
            else:
                factory, factory_kwargs = class_handler, None
    else:
        if using_aggregator:
            factory, factory_kwargs = aggregator_regression_handler, None
        else:
            factory, factory_kwargs = regression_handler, None
    loss_kwargs = dict() if loss_kwargs is None else dict(loss_kwargs)
    if factory_kwargs is not None:
        loss_kwargs.update(factory_kwargs)
    factory_signature = inspect.signature(factory)
    bad_keys = [k for k in loss_kwargs if k not in factory_signature.parameters]
    for k in bad_keys:
        del loss_kwargs[k]
    return partial(factory, **loss_kwargs)


def k_vs_k(predictions, target, k=20, num_samples=1000, pair_examples=None, feature_axes=-1):
    """
    Estimates how accurate a classifier would be if the classifier chose between
    1) the concatenated predictions of (e.g.) brain activity for the k examples corresponding to the true k examples
    and
    2) the concatenated predictions of k distractor examples
    by looking to see whether the vector formed by the k true or k distractor example predictions is closer to the
    vector formed by the k true target examples
    Args:
        predictions: The predictions output by a model. Should have shape (examples, ..., features) where examples is
            typically 1 per word, and features is typically voxels in fMRI or sensors in MEG. Accuracy is scored
            separately for each feature on the feature axis.
        target:  The true values for the features. Must have the same shape as predictions
        k: How many examples to combine together for the classifier
        num_samples: The number of samples of k-concatenations to use for estimating accuracy
        pair_examples: If present, must be a 1-D array with len(pair_examples) == len(predictions). pair_examples[i]
            gives the id of a group, or a negative value indicates that the word at index i should never be used. When
            a group id is given, the distractor will contain a word with the same group id at the position in the
            concatenated vector where word i is. Letting distractor_word_indices be the set of indices used in the
            distractor, and true_word_indices be the indices used in the true k examples, then:
            pair_examples[true_word_indices[j]] == pair_examples[distractor_word_indices[j]]
        feature_axes: An int or tuple indicating which axes to compute accuracy for. Axes which are neither the example
            axis (axis 0) or feature_axes will be used to make joint predictions.

    Returns:
        An accuracy array of shape
        (num_samples, predictions.shape[feature_axis_0], predictions.shape[feature_axis_1], ...)
    """

    if not np.array_equal(predictions.shape, target.shape):
        raise ValueError('predictions and target must have the same shape')

    if np.isscalar(feature_axes):
        feature_axes = [feature_axes]
    feature_axes = list(sorted([f if f > 0 else len(predictions.shape) + f for f in feature_axes]))
    transpose_axes = [i for i in range(len(predictions.shape)) if i not in feature_axes] + feature_axes
    if np.array_equal(transpose_axes, np.arange(len(predictions.shape))):
        transpose_axes = None
    if transpose_axes is not None:
        predictions = np.transpose(predictions, transpose_axes)
        target = np.transpose(target, transpose_axes)

    value_shape = predictions.shape[-len(feature_axes):]

    predictions = np.reshape(
        predictions, (predictions.shape[0], int(np.prod(predictions.shape[1:-len(feature_axes)])), -1,))
    target = np.reshape(
        target, (target.shape[0], int(np.prod(target.shape[1:-len(feature_axes)])), -1))

    # predictions, target data with the same shape: (words, ..., features)
    # k = how many words to classify at once
    # num_samples = how many words to classify
    accuracy = np.full((num_samples, target.shape[-1]), np.nan)

    if pair_examples is not None and len(pair_examples) > 0:
        if len(pair_examples) != len(predictions):
            raise ValueError('When specified, pair_examples must have 1 value per example')
        predictions = predictions[pair_examples >= 0]
        target = target[pair_examples >= 0]
        pair_examples = pair_examples[pair_examples >= 0]

    for index_sample in range(num_samples):
        indices_true = np.random.choice(len(target), k)
        sample_target = target[indices_true]
        sample_predictions_correct = predictions[indices_true]
        if pair_examples is not None and len(pair_examples) > 0:
            indices_distractor = _find_restricted_distractor_indices(indices_true, pair_examples)
        else:
            indices_distractor = np.random.choice(len(target), k)
        sample_predictions_incorrect = predictions[indices_distractor]

        sample_target = np.reshape(sample_target, (-1, sample_target.shape[-1]))
        sample_predictions_correct = np.reshape(
            sample_predictions_correct, (-1, sample_predictions_correct.shape[-1]))
        sample_predictions_incorrect = np.reshape(
            sample_predictions_incorrect, (-1, sample_predictions_incorrect.shape[-1]))

        distance_correct = np.sum((sample_target - sample_predictions_correct) ** 2, axis=0)
        distance_incorrect = np.sum((sample_target - sample_predictions_incorrect) ** 2, axis=0)
        accuracy[index_sample] = \
            (distance_correct < distance_incorrect) * 1.0 + (distance_correct == distance_incorrect) * 0.5

    return np.reshape(accuracy, (accuracy.shape[0],) + value_shape)


def _find_restricted_distractor_indices(indices_true, pair_examples):
    indices_distractor = np.zeros_like(indices_true)
    for i, w in enumerate(indices_true):
        id_group = pair_examples[w]
        other_words = np.where(pair_examples == id_group)[0]
        assert len(other_words) > 1
        indices_distractor[i] = np.random.permutation(np.setdiff1d(other_words, np.array(w)))[0]
    return indices_distractor


@dataclasses.dataclass
class FieldPredictions:
    predictions: np.ndarray
    masked_target: np.ndarray
    ids: np.ndarray
    word_ids: np.ndarray


def get_field_predictions(paths_obj, variation_set_name, field_names=None, index_run=None):
    (variation_set_name, _), settings = singleton_variation(variation_set_name)
    output_dir = os.path.join(paths_obj.result_path, variation_set_name, task_hash(settings))
    run_iterable = (index_run,) if index_run is not None else range(settings.num_runs)

    if field_names is None:
        aggregators = OrderedDict()
        is_single_field = False
    else:
        is_single_field = isinstance(field_names, str)
        if is_single_field:
            field_names = [field_names]
        aggregators = OrderedDict((field_name, Aggregator()) for field_name in field_names)

    for index_run in run_iterable:
        output_dir_run = os.path.join(output_dir, 'run_{}'.format(index_run))
        if not os.path.exists(os.path.join(output_dir_run, 'completed.txt')):
            raise ValueError('Incomplete results')
        validation_dir = os.path.join(output_dir_run, 'validation_predictions')
        if not os.path.exists(validation_dir):
            raise ValueError('Path does not exist: {}'.format(validation_dir))
        output_results = read_predictions(validation_dir, keys=field_names)
        for field_name in output_results:
            if field_name not in aggregators:
                aggregators[field_name] = Aggregator()
            for result in output_results[field_name]:
                aggregators[field_name].update(result, is_sequence=result.sequence_type != 'single')

    result = OrderedDict()
    for field_name in aggregators:
        target = np.array(aggregators[field_name].values('target'))
        mask = np.array(aggregators[field_name].values('mask'))
        result[field_name] = FieldPredictions(
            predictions=np.array(aggregators[field_name].values('prediction')),
            masked_target=np.where(mask, target, np.nan),
            ids=np.array(aggregators[field_name].values('unique_id')),
            word_ids=np.array(aggregators[field_name].values('word_ids')))

    if is_single_field:
        for field_name in result:
            return result[field_name]
    return result


def assemble_indexed_predictions(paths_obj, variation_set_name, index_run=None):
    field_predictions = get_field_predictions(paths_obj, variation_set_name, index_run=index_run)
    to_assemble = OrderedDict()
    for key in field_predictions:
        idx_under = key.rfind('_')
        if idx_under > 0:
            main_key = key[:idx_under]
            index_str = key[idx_under+1:]
            try:
                index = int(index_str)
                if main_key not in to_assemble:
                    to_assemble[main_key] = list()
                if index >= len(to_assemble[main_key]):
                    to_assemble[main_key].extend([None] * (index + 1 - len(to_assemble[main_key])))
                to_assemble[main_key][index] = field_predictions[key]
            except ValueError:
                to_assemble[key] = field_predictions[key]
        else:
            to_assemble[key] = field_predictions[key]
    result = OrderedDict()
    for key in to_assemble:
        if not isinstance(to_assemble[key], list):
            result[key] = to_assemble[key]
            continue
        for item in to_assemble[key]:
            assert(item is not None)
            indices_sort = np.argsort(item.ids, kind='mergesort')
            for f in dataclasses.fields(item):
                setattr(item, f.name, getattr(item, f.name)[indices_sort])
            assert(np.array_equal(item.ids, to_assemble[key][0].ids))
        assembled = dict()
        for f in dataclasses.fields(to_assemble[key][0]):
            if f.name == 'ids' or f.name == 'word_ids':
                assembled[f.name] = getattr(to_assemble[key][0], f.name)
            else:
                assembled[f.name] = np.concatenate(
                    list(np.expand_dims(getattr(item, f.name), 1) for item in to_assemble[key]), axis=1)
                if assembled[f.name].shape[-1] == 1:
                    assembled[f.name] = np.squeeze(assembled[f.name], axis=-1)
        result[key] = dataclasses.replace(to_assemble[key][0], **assembled)
    return result


def get_field_names(paths_obj, variation_set_name, index_run=None):
    (variation_set_name, _), settings = singleton_variation(variation_set_name)
    output_dir = os.path.join(paths_obj.result_path, variation_set_name, task_hash(settings))
    run_iterable = (index_run,) if index_run is not None else range(settings.num_runs)
    field_names = set()
    for index_run in run_iterable:
        output_dir_run = os.path.join(output_dir, 'run_{}'.format(index_run))
        if not os.path.exists(os.path.join(output_dir_run, 'completed.txt')):
            raise ValueError('Incomplete results')
        validation_dir = os.path.join(output_dir_run, 'validation_predictions')
        if not os.path.exists(validation_dir):
            raise ValueError('Path does not exist: {}'.format(validation_dir))
        field_names.update(get_output_keys(validation_dir))
    return list(sorted(field_names))


@dataclasses.dataclass
class StatisticsSpec:
    regression_metric: Optional[str] = None
    classifier_metric: Optional[str] = None
    test_type: str = 'ttest_rel'


@dataclasses.dataclass
class StatisticsResult:
    field_name: str
    metric_name: str
    test_type: str
    model_mean: np.ndarray
    model_std: np.ndarray
    baseline_mean: np.ndarray
    baseline_std: np.ndarray
    p_values: np.ndarray


def _statistics_helper(item):
    (variation_name, field_name, statistics_spec, index_run,
     k_vs_k_feature_axes, loss_handler_kwargs) = item

    (variation_set_name, training_variation_name), settings = singleton_variation(variation_name)
    paths_obj = _process_paths_obj

    run_iterable = [index_run] if index_run is not None else range(settings.num_runs)
    missing_runs = False

    aggregated = dict()
    metrics = dict()
    for index_run in run_iterable:
        output_dir = os.path.join(
            paths_obj.result_path, variation_set_name, task_hash(settings))
        if not os.path.exists(os.path.join(output_dir, 'run_{}'.format(index_run), 'completed.txt')):
            missing_runs = True
            continue
        validation_dir = os.path.join(output_dir, 'run_{}'.format(index_run), 'validation_predictions')
        output_results = read_predictions(validation_dir, keys=field_name)
        run_aggregated = Aggregator()
        loss = None
        is_active_loss = True
        for output_result in output_results:
            if not output_result.is_active_loss:
                is_active_loss = False
                break
            if loss is None:
                loss = output_result.critic_type
            else:
                assert (loss == output_result.critic_type)
            run_aggregated.update(output_result, is_sequence=output_result.sequence_type != 'single')

        if not is_active_loss:
            continue

        loss_handler_kwargs = dict(loss_handler_kwargs)
        if isinstance(k_vs_k_feature_axes, dict):
            if field_name in k_vs_k_feature_axes:
                loss_handler_kwargs['k_vs_k_feature_axes'] = k_vs_k_feature_axes[field_name]
            else:
                loss_handler_kwargs['k_vs_k_feature_axes'] = -1
        else:
            loss_handler_kwargs['k_vs_k_feature_axes'] = k_vs_k_feature_axes
        handler = make_prediction_handler(loss, loss_handler_kwargs)
        handler_result = handler(run_aggregated)
        is_many = isinstance(handler_result, list)
        if not is_many:
            handler_result = [handler_result]
            text_labels = [field_name]
        else:
            if variation_name not in _process_text_labels:
                _process_text_labels[variation_name] = TextLabels(paths_obj, variation_name, index_run)
            label_maker = _process_text_labels[variation_name]
            text_labels = label_maker.labels(field_name, len(handler_result))
        for text_label, result_dict in zip(text_labels, handler_result):
            if statistics_spec.regression_metric is not None and statistics_spec.regression_metric in result_dict:
                if text_label not in aggregated:
                    if statistics_spec.regression_metric == 'r':
                        # use 1 sample ttest with population mean = 0
                        aggregated[text_label] = (list(),)
                    else:
                        aggregated[text_label] = (list(), list())
                    metrics[text_label] = statistics_spec.regression_metric
                aggregated[text_label][0].append(result_dict[statistics_spec.regression_metric])
                if statistics_spec.regression_metric != 'r':
                    aggregated[text_label][1].append(
                        result_dict['mean_model_{}'.format(statistics_spec.regression_metric)])
            elif statistics_spec.classifier_metric is not None and statistics_spec.classifier_metric in result_dict:
                if text_label not in aggregated:
                    aggregated[text_label] = (list(), list())
                    metrics[text_label] = statistics_spec.classifier_metric
                aggregated[text_label][0].append(result_dict[statistics_spec.classifier_metric])
                aggregated[text_label][1].append(result_dict['m{}'.format(statistics_spec.classifier_metric)])

    result = dict()
    for text_label in aggregated:
        if len(aggregated[text_label]) == 2:
            model_values, baseline_values = aggregated[text_label]
            model_values = np.array(model_values)
            baseline_values = np.array(baseline_values)
            baseline_values = np.reshape(
                baseline_values, baseline_values.shape + (1,) * (len(model_values.shape) - len(baseline_values.shape)))
            if statistics_spec.test_type != 'ttest_rel':
                raise ValueError('Unknown test_type: {}'.format(statistics_spec.test_type))
            _, p_values = ttest_rel(model_values, baseline_values)
        else:
            model_values = aggregated[text_label][0]
            baseline_values = None
            if statistics_spec.test_type != 'ttest_rel':
                raise ValueError('Unknown test_type: {}'.format(statistics_spec.test_type))
            # shouldn't really assume the population mean here, but works for now since the only case is 'r'
            _, p_values = ttest_1samp(model_values, popmean=0)
        result[text_label] = StatisticsResult(
            text_label,
            statistics_spec.regression_metric,
            statistics_spec.test_type,
            np.mean(model_values, axis=0),
            np.std(model_values, axis=0),
            np.mean(baseline_values, axis=0) if baseline_values is not None else 0,
            np.std(baseline_values, axis=0) if baseline_values is not None else 0,
            p_values)

    return result, missing_runs


def compute_statistics_on_aggregated(
        paths_obj, variation_set_name, statistics_spec, field_names=None,
        index_run=None, k_vs_k_feature_axes=-1, **loss_handler_kwargs):

    if field_names is None:
        field_names = get_field_names(paths_obj, variation_set_name, index_run)

    task_arguments = [
        (variation_set_name, field_name, statistics_spec, index_run, k_vs_k_feature_axes, loss_handler_kwargs)
        for field_name in field_names]

    has_warned = False
    final_result = dict()
    with ProcessPoolExecutor(initializer=_set_process_paths_obj, initargs=(paths_obj,)) as ex:
        for result, is_missing_runs in tqdm(ex.map(_statistics_helper, task_arguments), total=len(task_arguments)):
            if is_missing_runs and not has_warned:
                print('Warning, some runs are missing')
                has_warned = True
            final_result.update(result)

    return final_result


def spatial_neighbor_edges(mask, order=1):

    def _make_indices():
        source_indices = np.arange(int(np.prod(mask.shape)))[np.reshape(mask, -1)]
        destination_indices = np.full(int(np.prod(mask.shape)), -1, dtype=source_indices.dtype)
        destination_indices[source_indices] = np.arange(len(source_indices))
        return np.reshape(destination_indices, mask.shape)

    offsets_1d = [np.arange(-order, order + 1)] * len(mask.shape)
    mesh = np.meshgrid(*offsets_1d)
    offsets = np.concatenate(list(np.reshape(m, (-1, 1)) for m in mesh), axis=1)
    offsets = offsets[np.logical_not(np.all(offsets == 0, axis=1))]

    indices = _make_indices()
    padded_indices = np.pad(indices, order, mode='constant', constant_values=-1)
    unpad_slices = tuple(slice(order, -order) for _ in range(len(mask.shape)))

    starts = list()
    ends = list()
    for offset in offsets:
        neighbors = np.roll(padded_indices, offset, np.arange(np.ndim(mask)))[unpad_slices][mask]
        indicator_valid = neighbors >= 0
        starts.append(np.arange(len(neighbors))[indicator_valid])
        ends.append(neighbors[indicator_valid])

    return np.concatenate(starts), np.concatenate(ends)
