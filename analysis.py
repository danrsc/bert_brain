import os
import warnings
from functools import partial
from collections import OrderedDict
import dataclasses
from typing import Tuple, Optional
import inspect
import fnmatch

import numpy as np
from scipy.misc import logsumexp

from run_variations import task_hash, named_variations
from bert_erp_modeling import CriticMapping
from result_output import read_predictions, read_loss_curve
from text_grid import TextGrid, TextWrapStyle, write_text_grid_to_console


output_order = (
    'mse',       # mean squared error
    'pove',      # proportion of variance explained
    'povu',      # proportion of variance unexplained
    'variance',
    'r_seq',     # avg (over batch) of sequence correlation values (i.e. correlation within a sequence)
    'xent',      # cross entropy
    'acc',       # accuracy
    'macc',      # mode accuracy - the accuracy one would get if one picked the mode
    'poma',      # proportion of mode accuracy; < 1 is bad
    'prec',      # precision
    'rec',       # recall
    'f1')


class Aggregator:
    def __init__(self):
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
            elif np.isscalar(result[field]):
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

    def counts(self, name=None):
        return self._counts[name]


def read_variation_results(paths, variation_set_name, training_variation, aux_loss, num_runs,
                           compute_scalar=True, **loss_handler_kwargs):
    output_dir = os.path.join(paths.result_path, variation_set_name, task_hash(set(training_variation)))
    aggregated = dict()
    losses = dict()
    count_runs = 0
    has_warned = False
    for index_run in range(num_runs):
        run_aggregated = dict()
        validation_npz_path = os.path.join(output_dir, 'run_{}'.format(index_run), 'output_validation.npz')
        if not os.path.exists(validation_npz_path):
            if not has_warned:
                print('Warning: results incomplete. Some output files not found')
            has_warned = True
            continue
        count_runs += 1
        output_results_by_name = read_predictions(validation_npz_path)
        for name in output_results_by_name:
            if name not in training_variation and name not in aux_loss:
                continue
            output_results = output_results_by_name[name]
            for output_result in output_results:
                if name not in run_aggregated:
                    run_aggregated[name] = Aggregator()
                if name not in losses:
                    losses[name] = output_result.critic_type
                else:
                    assert (losses[name] == output_result.critic_type)
                run_aggregated[name].update(output_result, is_sequence=output_result.sequence_type != 'single')
        for name in run_aggregated:
            handler = make_prediction_handler(losses[name], loss_handler_kwargs)
            result_dict = handler(run_aggregated[name])
            if compute_scalar:
                result_dict = dict((k, np.nanmean(result_dict[k])) for k in result_dict)
            if name not in aggregated:
                aggregated[name] = Aggregator()
            aggregated[name].update(result_dict, is_sequence=False)

    return aggregated, count_runs


def print_variation_results_sliced(
        paths, variation_set_name, training_variation, aux_loss, num_runs, metric='pove',
        field_precision=2, num_values_per_table=10, **loss_handler_kwargs):

    aggregated, count_runs = read_variation_results(paths, variation_set_name, training_variation, aux_loss, num_runs,
                                                    compute_scalar=False, **loss_handler_kwargs)

    values = OrderedDict((name, np.nanmean(aggregated[name].values(metric), axis=0)) for name in aggregated)

    grouped_by_shape = OrderedDict()
    for name in values:
        if values[name].shape not in grouped_by_shape:
            grouped_by_shape[values[name].shape] = [name]
        else:
            grouped_by_shape[values[name].shape].append(name)

    print('Variation ({} of {} runs found): {}'.format(count_runs, num_runs, ', '.join(sorted(training_variation))))

    for shape in grouped_by_shape:
        num_tables = int(np.ceil(np.prod(shape) / num_values_per_table))
        for i in range(num_tables):
            indices = np.arange(num_values_per_table) + i * num_values_per_table
            indices = indices[indices < np.prod(shape)]
            indices = np.unravel_index(indices, shape)

            text_grid = TextGrid()
            text_grid.append_value('name', column_padding=2)
            # indices is a tuple of arrays, length 1 is a special case
            for index in indices[0] if len(indices) == 1 else zip(indices):
                text_grid.append_value('{}'.format(index), line_style=TextWrapStyle.right_justify, column_padding=2)
            text_grid.next_row()
            value_format = '{' + ':.{}f'.format(field_precision) + '}'
            for name in grouped_by_shape[shape]:
                text_grid.append_value(name, column_padding=2)
                current_values = values[name][indices]
                for value in current_values:
                    text_grid.append_value(
                        value_format.format(value), line_style=TextWrapStyle.right_justify, column_padding=2)
                text_grid.next_row()

            write_text_grid_to_console(text_grid, width='tight')
            print('')

    print('')
    print('')


def print_variation_results(paths, variation_set_name, training_variation, aux_loss, num_runs, field_precision=2,
                            **loss_handler_kwargs):

    aggregated, count_runs = read_variation_results(paths, variation_set_name, training_variation, aux_loss, num_runs,
                                                    **loss_handler_kwargs)

    metrics = list()
    for metric in output_order:
        if any(metric in aggregated[name] for name in aggregated):
            metrics.append(metric)

    text_grid = TextGrid()
    text_grid.append_value('name', column_padding=2)
    for metric in metrics:
        text_grid.append_value(metric, line_style=TextWrapStyle.right_justify, column_padding=2)
    text_grid.next_row()
    value_format = '{' + ':.{}f'.format(field_precision) + '}'
    for name in aggregated:
        text_grid.append_value(name, column_padding=2)
        for metric in metrics:
            value = np.nanmean(aggregated[name].values(metric)) if metric in aggregated[name] else np.nan
            text_grid.append_value(value_format.format(value), line_style=TextWrapStyle.right_justify, column_padding=2)
        text_grid.next_row()

    print('Variation ({} of {} runs found): {}'.format(count_runs, num_runs, ', '.join(sorted(training_variation))))
    write_text_grid_to_console(text_grid, width='tight')
    print('')
    print('')


def sentence_predictions(paths, variation_set_name, training_variation, aux_loss, num_runs):
    output_dir = os.path.join(paths.result_path, variation_set_name, task_hash(set(training_variation)))
    result = dict()
    has_warned = False
    for index_run in range(num_runs):
        validation_npz_path = os.path.join(output_dir, 'run_{}'.format(index_run), 'output_validation.npz')
        if not os.path.exists(validation_npz_path):
            if not has_warned:
                print('warning: results are incomplete. Some runs not found')
            has_warned = True
            continue
        output_results = np.load(validation_npz_path)
        for output_result in output_results:
            name = output_result.name
            if name not in training_variation and name not in aux_loss:
                continue
            data_key, unique_id = output_result.data_key, output_result.unique_id
            if data_key not in result:
                result[data_key] = dict()
            if unique_id not in result[data_key]:
                result[data_key][unique_id] = dict()
            if name not in result[data_key][unique_id]:
                result[data_key][unique_id][name] = list()
            result[data_key][unique_id][name].append(output_result)

    return result


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


def aggregator_regression_handler(aggregator, k_vs_k_num_samples=0, k_vs_k_k=20):
    target = np.array(aggregator.values('target'))
    predictions = np.array(aggregator.values('prediction'))
    mask = np.array(aggregator.values('mask'))

    target_counts = np.array(aggregator.counts('target'))
    prediction_counts = np.array(aggregator.counts('prediction'))
    assert(np.array_equal(target_counts, prediction_counts))

    splits = None
    if np.any(target_counts > 1):
        splits = np.cumsum(target_counts)[:-1]

    return regression_handler(predictions, target, mask, k_vs_k_num_samples, k_vs_k_k, splits, is_single_example=False)


def regression_handler(
        predictions, target, mask, k_vs_k_num_samples=0, k_vs_k_k=20, splits=None, is_single_example=False):

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
    mse = np.nanmean(np.square(predictions - masked_target), axis=0)

    variance = np.where(variance < 1e-8, np.nan, variance)

    result = dict(
        mse=mse,
        pove=1 - (mse / variance),
        povu=(mse / variance),
        variance=variance,
        r_seq=seq_r)

    if k_vs_k_num_samples > 0:
        k_vs_k_mask = np.reshape(mask, (mask.shape[0], -1))
        if not np.all(k_vs_k_mask == k_vs_k_mask[:, 0:1]):
            raise ValueError('For k_vs_k, the mask must be the same for all features')
        k_vs_k_mask = k_vs_k_mask[:, 0]
        accuracy = k_vs_k(
            predictions[k_vs_k_mask], target[k_vs_k_mask], k=k_vs_k_k, num_samples=k_vs_k_num_samples)
        result['{0}_vs_{0}'.format(k_vs_k_k)] = np.mean(accuracy, axis=0)

    return result


def bincount_axis(x, weights=None, minlength=None, axis=-1):

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


def aggregator_class_handler(aggregator, pos_weight=None, is_binary=False):

    target = np.array(aggregator.values('target'))
    predictions = np.array(aggregator.values('prediction'))
    mask = np.array(aggregator.values('mask'))

    return class_handler(predictions, target, mask, pos_weight, is_binary)


def class_handler(predictions, target, mask, pos_weight=None, is_binary=False, is_single_example=False):
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
        accuracy = (np.sum(true_positive, axis=0) + np.sum(true_negative, axis=0)) / np.sum(indicator_valid, axis=0)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            pos_acc = np.sum(target_positive, axis=0) / np.sum(indicator_valid, axis=0)
            neg_acc = np.sum(target_negative, axis=0) / np.sum(indicator_valid, axis=0)
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

        poma = accuracy / mode_accuracy

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
        log_sum = np.reshape(log_sum, log_sum.shape + (1,))
        log_softmax = predictions - log_sum
        predicted_class = np.argmax(predictions, axis=-1)
        if is_hard_label:
            log_softmax = np.reshape(log_softmax, (-1, log_softmax.shape[-1]))
            indices = np.reshape(target, -1)
            cross_entropy = np.zeros(indices.shape, log_softmax.dtype)
            for result_index, (index, current) in enumerate(zip(indices, log_softmax)):
                if index < 0:
                    cross_entropy[result_index] = np.nan
                else:
                    cross_entropy[result_index] = log_softmax[index]
            cross_entropy = np.reshape(cross_entropy, target.shape)
            cross_entropy = np.nanmean(cross_entropy, axis=0)
            accuracy = np.sum(np.equal(predicted_class, target), axis=0) / np.sum(np.greater_equal(target, 0), axis=0)
            modes = np.argmax(bincount_axis(target, axis=0), axis=0)
            mode_accuracy = np.sum(np.equal(modes, target), axis=0) / np.sum(np.greater_equal(target, 0), axis=0)
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

        return dict(xent=cross_entropy, acc=accuracy, macc=mode_accuracy, poma=poma)


_prediction_handlers = dataclasses.asdict(CriticMapping(
    mse=aggregator_regression_handler,
    k_least_se=aggregator_regression_handler,
    pearson=aggregator_regression_handler,
    cross_entropy=aggregator_class_handler,
    binary_cross_entropy=(aggregator_class_handler, dict(is_binary=True)),
    soft_label_cross_entropy=aggregator_class_handler,
    single_mse=aggregator_regression_handler,
    single_k_least_se=aggregator_regression_handler,
    single_cross_entropy=aggregator_class_handler,
    single_binary_cross_entropy=(aggregator_class_handler, dict(is_binary=True)),
    single_soft_label_cross_entropy=aggregator_class_handler), dict_factory=OrderedDict)


_no_aggregator_prediction_handlers = dataclasses.asdict(CriticMapping(
    mse=regression_handler,
    k_least_se=regression_handler,
    pearson=regression_handler,
    cross_entropy=class_handler,
    binary_cross_entropy=(class_handler, dict(is_binary=True)),
    soft_label_cross_entropy=class_handler,
    single_mse=regression_handler,
    single_k_least_se=regression_handler,
    single_cross_entropy=class_handler,
    single_binary_cross_entropy=(class_handler, dict(is_binary=True)),
    single_soft_label_cross_entropy=class_handler), dict_factory=OrderedDict)


def make_prediction_handler(which_loss, loss_kwargs=None, using_aggregator=True):
    handler_map = _prediction_handlers if using_aggregator else _no_aggregator_prediction_handlers
    if which_loss not in handler_map:
        raise ValueError('Unknown value for which_loss. Known values are: {}'.format(handler_map.keys()))
    factory = handler_map[which_loss]
    loss_kwargs = {} if loss_kwargs is None else dict(loss_kwargs)
    if isinstance(factory, tuple):
        factory, factory_kwargs = factory
        loss_kwargs.update(factory_kwargs)
    factory_signature = inspect.signature(factory)
    bad_keys = [k for k in loss_kwargs if k not in factory_signature.parameters]
    for k in bad_keys:
        del loss_kwargs[k]
    return partial(factory, **loss_kwargs)


def k_vs_k(predictions, target, k=20, num_samples=1000, pair_examples=None):
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

    Returns:
        An accuracy array of shape (num_samples, features)
    """

    if not np.array_equal(predictions.shape, target.shape):
        raise ValueError('predictions and target must have the same shape')

    value_shape = predictions.shape[1:]
    predictions = np.reshape(predictions, (predictions.shape[0], -1))
    target = np.reshape(target, (target.shape[0], -1))

    # predictions, target data with the same shape: (words, ..., features)
    # k = how many words to classify at once
    # num_samples = how many words to classify
    accuracy = np.full((num_samples, target.shape[1]), np.nan)

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


def text_heat_map_html(words, scores, vmin=None, vmax=None, cmap=None, text_color=None):
    from matplotlib import cm, colors
    cmap = cm.ScalarMappable(colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    fmt = '<span style="background-color:{hex}{text_color}">{word}</span>'
    fmt = fmt.format(hex='{hex}', word='{word}', text_color='' if text_color is None else ';color:{text_color}')
    word_colors = cmap.to_rgba(scores)
    return '&nbsp;'.join(
        [fmt.format(word=w, hex=colors.to_hex(c), text_color=text_color) for w, c in zip(words, word_colors)])


def average_unique_steps_within_loss_curves(curves):
    for curve in curves:
        unique_steps = np.unique(curve.steps)
        step_values = list()
        step_epochs = list()
        for step in unique_steps:
            step_values.append(np.nanmean(curve.values[curve.steps == step]))
            step_epochs.append(curve.epochs[curve.steps == step][0])
        curve.steps = unique_steps
        curve.epochs = np.array(step_epochs)
        curve.values = np.array(step_values)


def average_unique_epochs_within_loss_curves(curves):
    for curve in curves:
        unique_epochs = np.unique(curve.epochs)
        epoch_values = list()
        for epoch in unique_epochs:
            epoch_values.append(np.nanmean(curve.values[curve.epochs == epoch]))
        curve.steps = unique_epochs
        curve.epochs = unique_epochs
        curve.values = np.array(epoch_values)


@dataclasses.dataclass
class LossCurve:
    training_variation: Tuple[str, ...]
    train_eval_kind: str
    index_run: int
    key: str
    epochs: np.ndarray
    steps: np.ndarray
    values: np.ndarray


def loss_curves_for_variation(paths, variation_set_name):
    training_variations, _, num_runs, _, _ = named_variations(variation_set_name)

    def read_curve(kind, training_variation_, index_run_):
        file_name = 'train_curve.npz' if kind == 'train' else 'validation_curve.npz'
        output_dir = os.path.join(paths.result_path, variation_set_name, task_hash(set(training_variation_)))
        curve_path = os.path.join(output_dir, 'run_{}'.format(index_run_), file_name)
        result_ = list()
        if os.path.exists(curve_path):
            curve = read_loss_curve(curve_path)
            for key in curve:
                result_.append(
                    LossCurve(training_variation_, kind, index_run_, key, curve[key][0], curve[key][1], curve[key][2]))
        return result_

    result = list()
    for training_variation in training_variations:
        for index_run in range(num_runs):
            result.extend(read_curve('train', training_variation, index_run))
            result.extend(read_curve('validation', training_variation, index_run))

    return result


@dataclasses.dataclass
class ResultQuery:
    variant_set_name: str
    metric: str
    key: str
    training_variation: Optional[str] = None
    second_variation_set_name: Optional[str] = None
    second_training_variation: Optional[str] = None


def query_results(paths, result_queries, compute_scalar=False, **loss_handler_kwargs):
    cache = dict()
    result = list()
    for result_query in result_queries:
        training_variations, _, num_runs, _, aux_loss = named_variations(result_query.variation_set_name)
        query_training_variation = set(result_query.training_variation) \
            if result_query.training_variation is not None else None
        for training_variation in training_variations:
            training_variation = set(training_variation)
            if query_training_variation is None or query_training_variation == training_variation:
                cache_key = result_query.variation_set_name, training_variation
                if cache_key not in cache:
                    cache[cache_key] = read_variation_results(
                        paths, result_query.variation_set_name, training_variation, aux_loss, num_runs,
                        compute_scalar=compute_scalar, **loss_handler_kwargs)
                aggregated, count_runs = cache[cache_key]
                for aggregated_key in aggregated:
                    if fnmatch.fnmatch(aggregated_key, result_query.key):
                        result_query = dataclasses.replace(result_query, key=aggregated_key)
                        if result_query.metric == 'k_vs_k':
                            values = None
                            for metric in aggregated[result_query.key]:
                                split_metric = metric.split('_')
                                if len(split_metric) == 3 \
                                        and split_metric[1] == 'vs' and split_metric[0] == split_metric[2]:
                                    values = aggregated[result_query.key].values(metric)
                                    break
                            if values is None:
                                raise ValueError('k_vs_k not found')
                        else:
                            values = aggregated[result_query.key].values(result_query.metric)
                        if result_query.training_variation is None:
                            result_query = dataclasses.replace(result_query, training_variation=training_variation)
                        result.append((result_query, values))
    for idx_result in range(len(result)):
        if result[idx_result][0].second_variation_set_name is not None:
            result_query, first_values = result[idx_result]
            training_variations, _, num_runs, _, aux_loss = named_variations(result_query.second_variation_set_name)
            if result_query.second_training_variation is None:
                result_query = dataclasses.replace(
                    result_query, second_training_variation=result_query.training_variation)
            query_training_variation = set(result_query.second_training_variation)
            for training_variation in training_variations:
                if query_training_variation == training_variation:
                    cache_key = result_query.second_variation_set_name, training_variation
                    if cache_key not in cache:
                        cache[cache_key] = read_variation_results(
                            paths, result_query.second_variation_set_name, training_variation, aux_loss, num_runs,
                            compute_scalar=compute_scalar, **loss_handler_kwargs)
                    aggregated, count_runs = cache[cache_key]
                    if result_query.metric == 'k_vs_k':
                        second_values = None
                        for metric in aggregated[result_query.key]:
                            split_metric = metric.split('_')
                            if len(split_metric) == 3 \
                                    and split_metric[1] == 'vs' and split_metric[0] == split_metric[2]:
                                second_values = aggregated[result_query.key].values(metric)
                                break
                        if second_values is None:
                            raise ValueError('k_vs_k not found')
                    else:
                        second_values = aggregated[result_query.key].values(result_query.metric)
                    result[idx_result] = (result_query, first_values, second_values)
                    break  # break out of looping over training variations
    return result
