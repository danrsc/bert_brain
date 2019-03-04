import os
import json
import warnings
from functools import partial

import numpy as np
from scipy.misc import logsumexp

from run_regression import task_hash
from bert_erp_common import SwitchRemember


output_order = ('mse', 'pove', 'povu', 'variance', 'r_seq', 'xent', 'acc', 'macc', 'poma', 'prec', 'rec', 'f1')


def print_variation_results(paths, variation_set_name, training_variation, aux_loss, num_runs):
    output_dir = os.path.join(paths.base_path, 'bert', variation_set_name, task_hash(set(training_variation)))
    aggregated = dict()
    losses = dict()
    for index_run in range(num_runs):
        run_aggregated = dict()
        validation_json_path = os.path.join(output_dir, 'run_{}'.format(index_run), 'output_validation.json')
        with open(validation_json_path, 'rt') as validation_json_file:
            output_results = json.load(validation_json_file)
        for output_result in output_results:
            name = output_result['name']
            if name not in training_variation and name not in aux_loss:
                continue
            if name not in run_aggregated:
                run_aggregated[name] = (list(), list(), list())
            if name not in losses:
                losses[name] = output_result['critic_type']
            else:
                assert(losses[name] == output_result['critic_type'])
            if np.isscalar(output_result['target']):
                run_aggregated[name][0].append(output_result['prediction'])
                run_aggregated[name][1].append(output_result['target'])
            else:
                run_aggregated[name][0].extend(output_result['prediction'])
                run_aggregated[name][1].extend(output_result['target'])
            if output_result['mask'] is not None:
                if np.isscalar(output_result['mask']):
                    run_aggregated[name][2].append(output_result['mask'])
                else:
                    run_aggregated[name][2].extend(output_result['mask'])
        for name in run_aggregated:
            predictions = np.array(run_aggregated[name][0])
            target = np.array(run_aggregated[name][1])
            if len(run_aggregated[name][2]) == 0:
                mask = None
            else:
                assert(len(run_aggregated[name][2]) == len(run_aggregated[name][1]))
                mask = np.array(run_aggregated[name][2])
            handler = make_prediction_handler(losses[name])
            result_dict = handler(mask, predictions, target)
            if name not in aggregated:
                aggregated[name] = dict()
            for metric in result_dict:
                if metric not in aggregated[name]:
                    aggregated[name][metric] = list()
                aggregated[name][metric].append(result_dict[metric])

    metrics = list()
    for metric in output_order:
        if any(metric in aggregated[name] for name in aggregated):
            metrics.append(metric)

    header_format = '{name:8}'
    row_format = '{name:8}'
    for metric in metrics:
        header_format = header_format + '  {' + metric + ':>10}'
        row_format = row_format + '  {' + metric + ':>10.6}'

    print('Variation: {}'.format(', '.join(sorted(training_variation))))
    print(header_format.format(name='name', **dict((m, m) for m in metrics)))
    for name in aggregated:
        print(row_format.format(
            name=name, **dict(
                (m, np.nanmean(aggregated[name][m]) if m in aggregated[name] else np.nan) for m in metrics)))
    print('')
    print('')


def sentence_predictions(paths, variation_set_name, training_variation, aux_loss, num_runs):
    output_dir = os.path.join(paths.base_path, 'bert', variation_set_name, task_hash(set(training_variation)))
    result = dict()
    for index_run in range(num_runs):
        validation_json_path = os.path.join(output_dir, 'run_{}'.format(index_run), 'output_validation.json')
        with open(validation_json_path, 'rt') as validation_json_file:
            output_results = json.load(validation_json_file)
        for output_result in output_results:
            name = output_result['name']
            if name not in training_variation and name not in aux_loss:
                continue
            data_key, unique_id = output_result['data_key'], output_result['unique_id']
            if data_key not in result:
                result[data_key] = dict()
            if unique_id not in result[data_key]:
                result[data_key][unique_id] = dict()
            if name not in result[data_key][unique_id]:
                result[data_key][unique_id][name] = list()
            result[data_key][unique_id][name].append(output_result)

    return result


def nan_pearson(x, y, axis=1, keepdims=False):
    if not np.array_equal(x.shape, y.shape):
        raise ValueError('x and y must be the same shape')
    if np.prod(x.shape) == 0:
        return np.nan
    x = x - np.nanmean(x, axis=axis, keepdims=True)
    y = y - np.nanmean(y, axis=axis, keepdims=True)
    with warnings.catch_warnings():
        # ddof < 1 for slice
        warnings.filterwarnings('ignore', category=RuntimeWarning)
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


def regression_handler(mask, predictions, target):
    masked_target = np.where(mask, target, np.nan) if mask is not None else target
    variance = np.nanvar(masked_target)
    mse = np.nanmean(np.square(predictions - masked_target))
    return dict(
        mse=mse,
        pove=1 - (mse / variance),
        povu=mse / variance,
        variance=variance,
        r_seq=np.nanmean(nan_pearson(predictions, masked_target), axis=0))


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


def class_handler(mask, predictions, target, pos_weight=None, is_binary=False):

    target = np.where(mask, target, np.nan) if mask is not None else target

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

        target_positive = np.logical_and(np.greater(target, 0), indicator_valid)
        target_negative = np.logical_and(np.equal(target, 0), indicator_valid)
        predictions_positive = np.logical_and(np.greater_equal(predictions, 0), indicator_valid)
        predictions_negative = np.logical_and(np.less(predictions, 0), indicator_valid)

        true_positive = np.logical_and(predictions_positive, target_positive)
        true_negative = np.logical_and(predictions_negative, target_negative)
        false_positive = np.logical_and(predictions_positive, target_negative)
        false_negative = np.logical_and(predictions_negative, target_positive)

        precision = np.sum(true_positive, axis=0) / (np.sum(true_positive, axis=0) + np.sum(false_positive, axis=0))
        # nothing was classified as positive, define this to be precision 0.
        precision = np.where(np.sum(true_positive, axis=0) + np.sum(false_positive, axis=0) == 0, 0., precision)
        recall = np.sum(true_positive, axis=0) / (np.sum(true_positive, axis=0) + np.sum(false_negative, axis=0))
        # there are no real positive examples, define this to be recall 1.
        recall = np.where(np.sum(true_positive, axis=0) + np.sum(false_negative, axis=0) == 0, 1., recall)
        accuracy = (np.sum(true_positive, axis=0) + np.sum(true_negative, axis=0)) / np.sum(indicator_valid, axis=0)
        pos_acc = np.sum(target_positive, axis=0) / np.sum(indicator_valid, axis=0)
        neg_acc = np.sum(target_negative, axis=0) / np.sum(indicator_valid, axis=0)
        positive_better = np.sum(np.greater_equal(pos_acc, neg_acc)) > np.sum(np.less(pos_acc, neg_acc))
        mode_accuracy = pos_acc if positive_better else neg_acc
        f1 = 2 * precision * recall / (precision + recall)
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


def make_prediction_handler(which_loss, loss_kwargs=None):
    which_loss = SwitchRemember(which_loss)
    if loss_kwargs is None:
        loss_kwargs = {}
    if which_loss == 'mse':
        return partial(regression_handler, **loss_kwargs)
    elif which_loss == 'pearson':
        return partial(regression_handler, **loss_kwargs)
    elif which_loss == 'cross_entropy':
        return partial(class_handler, **loss_kwargs)
    elif which_loss == 'binary_cross_entropy':
        return partial(class_handler, is_binary=True, **loss_kwargs)
    elif which_loss == 'soft_label_cross_entropy':
        return partial(class_handler, **loss_kwargs)
    elif which_loss == 'sequence_cross_entropy':
        return partial(class_handler, **loss_kwargs)
    elif which_loss == 'sequence_binary_cross_entropy':
        return partial(class_handler, is_binary=True, **loss_kwargs)
    elif which_loss == 'sequence_soft_label_cross_entropy':
        return partial(class_handler, **loss_kwargs)
    else:
        raise ValueError('Unknown value for which_loss. Known values are: {}'.format(which_loss.tests))
