import os
from collections import OrderedDict
from dataclasses import dataclass
import pickle
import torch
import gc

import numpy as np
from scipy.stats import ttest_1samp
from tqdm.auto import tqdm

from .modeling import BertMultiPredictionHead, KeyedSingleTargetSpanMaxPool, PooledFromKTokens
from .data_sets import CorpusDatasetFactory, DataIdMultiDataset
from .experiments import singleton_variation, task_hash


__all__ = [
    'load_existing_data_sets',
    'get_model_weights',
    'write_pairwise_metric_statistics',
    'filter_pairwise_metric_statistics',
    'TextLabels']


def load_existing_data_sets(paths_obj, variation_name, index_run):
    _, settings = singleton_variation(variation_name)
    corpus_dataset_factory = CorpusDatasetFactory(cache_path=paths_obj.cache_path)
    data_set_paths = list()
    for corpus in settings.corpora:
        data_set_paths.append(corpus_dataset_factory.maybe_make_data_set_files(
            index_run,
            corpus,
            settings.preprocessors,
            settings.get_split_function(corpus, index_run),
            settings.preprocess_fork_fn,
            False,
            paths_obj,
            settings.max_sequence_length,
            settings.create_meta_train_dataset,
            paths_only=True))

    train_data, validation_data, test_data, meta_train_data = (
        DataIdMultiDataset(
            which,
            data_set_paths,
            settings.all_loss_tasks,
            data_id_in_batch_keys=settings.data_id_in_batch_keys,
            filter_when_not_in_loss_keys=settings.filter_when_not_in_loss_keys,
            field_spec_replacers=settings.field_spec_replacers)
        for which in ('train', 'validation', 'test', 'meta_train'))

    return train_data, validation_data, test_data, meta_train_data


class TextLabels:

    def __init__(self, paths_obj, variation_name, index_run):
        train_data, _, _, _ = load_existing_data_sets(paths_obj, variation_name, index_run)
        self.train_data = train_data
        self.response_field_to_id = dict()
        for index, name in enumerate(train_data.response_fields):
            self.response_field_to_id[name] = index

    def labels(self, output_name, num_items):
        if num_items == 1:
            return [output_name]
        text_labels = None
        if output_name in self.response_field_to_id:
            text_labels = self.train_data.text_labels_for_field(output_name)
            if text_labels is not None and len(text_labels) != num_items:
                raise ValueError('Unexpected number of items. Expected {}, got {}'.format(len(text_labels), num_items))
        if text_labels is None:
            text_labels = ['{}'.format(i) for i in range(num_items)]
        assert (len(text_labels) == num_items)
        return list('{}.{}'.format(output_name, label) for label in text_labels)


def get_model_weights(paths_obj, variation_name, graph_parts_to_compare, index_run=None):

    _, settings = singleton_variation(variation_name)

    if index_run is not None:
        runs = [index_run] if np.ndim(index_run) == 0 else index_run
    else:
        runs = range(settings.num_runs)

    index_runs = list()
    ordered_names = None
    output = list()
    token_weight_output = None

    label_maker = None

    for index_run in runs:
        model_dir = os.path.join(paths_obj.model_path, variation_name, task_hash(settings), 'run_{}'.format(index_run))
        if not os.path.exists(model_dir):
            continue

        # assume that the names, which is all we need the dataset for, are consistent across runs
        if label_maker is None:
            label_maker = TextLabels(paths_obj, variation_name, index_run)

        model = BertMultiPredictionHead.from_pretrained(model_dir)
        rationalized_output_weights = OrderedDict()
        local_token_weight_output = OrderedDict()
        for k in model.prediction_head.head_graph_parts:
            if k in graph_parts_to_compare:
                if isinstance(model.prediction_head.head_graph_parts[k], PooledFromKTokens):
                    local_token_weight_output[k] = model.prediction_head.head_graph_parts[k].get_token_weights()
                    continue

                output_weights = model.prediction_head.head_graph_parts[k].get_output_weights()
                for output_name in output_weights:
                    w = output_weights[output_name]
                    if np.ndim(w) > 2:
                        if np.any(np.logical_not(np.equal(w.shape[:-2], 1))):
                            raise ValueError('Unable to handle data shape')
                        w = np.reshape(w, w.shape[-2:])
                    if np.ndim(w) > 1:
                        if w.shape[0] == 1:
                            w = [np.squeeze(w, 0)]
                        else:
                            w = list(np.squeeze(w_item, 0) for w_item in np.split(w, len(w)))
                    else:
                        w = [w]

                    text_labels = label_maker.labels(output_name, len(w))
                    if isinstance(model.prediction_head.head_graph_parts[k], KeyedSingleTargetSpanMaxPool) \
                            and model.prediction_head.head_graph_parts[k].num_spans > 1:
                        for w_item, text_label in zip(w, text_labels):
                            for index_span, w_span in enumerate(
                                    np.split(w_item, model.prediction_head.head_graph_parts[k].num_spans)):
                                rationalized_output_weights['{}.span{}'.format(text_label, index_span)] = (
                                    label_maker.response_field_to_id[output_name], w_span)
                    else:
                        for w_item, text_label in zip(w, text_labels):
                            if output_name not in label_maker.response_field_to_id:
                                sort_val = -1
                            else:
                                sort_val = label_maker.response_field_to_id[output_name]
                            rationalized_output_weights[text_label] = (sort_val, w_item)

        rationalized_output_weights = OrderedDict(
            (k, rationalized_output_weights[k][1])
            for i, k in sorted(
                enumerate(rationalized_output_weights),
                key=lambda ik: (rationalized_output_weights[ik[1]][0], ik[0])))

        ordered_names_ = list()
        ordered_weights = list()
        for i, k in enumerate(rationalized_output_weights):
            ordered_names_.append(k)
            ordered_weights.append(rationalized_output_weights[k])
        if ordered_names is not None:
            if len(ordered_names_) != len(ordered_names):
                raise ValueError('Inconsistent weights over runs')
            for a, b in zip(ordered_names_, ordered_names):
                if a != b:
                    raise ValueError('Inconsistent weights over runs')
        else:
            ordered_names = ordered_names_
        ordered_weights = np.array(ordered_weights)

        if index_run == 0:
            if len(local_token_weight_output) > 0:
                token_weight_output = OrderedDict(
                    (k, list([local_token_weight_output[k]])) for k in local_token_weight_output)
        else:
            if len(local_token_weight_output) > 0 and token_weight_output is None:
                raise ValueError('Inconsistent token weights over runs')
            if len(local_token_weight_output) == 0 and token_weight_output is not None:
                raise ValueError('Inconsistent token weights over runs')
            if token_weight_output is not None and len(local_token_weight_output) != len(token_weight_output):
                raise ValueError('Inconsistent token weights over runs')
            for k in local_token_weight_output:
                if k not in token_weight_output:
                    raise ValueError('Inconsistent token weights over runs')
                token_weight_output[k].append(local_token_weight_output[k])

        index_runs.append(index_run)
        output.append(ordered_weights)

    del model
    # model is somehow getting put onto cuda; need to fix, but at least release it here
    gc.collect()
    torch.cuda.empty_cache()

    if token_weight_output is not None:
        token_weight_output = OrderedDict((k, np.concatenate(token_weight_output[k])) for k in token_weight_output)
        return ordered_names, index_runs, np.array(output), token_weight_output

    return ordered_names, index_runs, np.array(output)


def cosine_similarity(w_1, w_2):
    return np.dot(w_1, w_2) / (np.linalg.norm(w_1) * np.linalg.norm(w_2))


def _upper_triangle_iterate(num_items):
    for i in range(num_items):
        for j in range(i + 1, num_items):
            yield i, j


@dataclass
class PairwiseMetricStatistics:
    name_1: str
    name_2: str
    mu: float
    sigma: float
    t_stat: float
    p_value: float
    num_samples: int


def write_pairwise_metric_statistics(output_file_path, names, weights, metric_fn=cosine_similarity, population_mean=0):
    with open(output_file_path, 'wb') as output_file:
        num_pairs = len(names) * (len(names) - 1) // 2
        pickle.dump(num_pairs, output_file, protocol=pickle.HIGHEST_PROTOCOL)
        for i, j in tqdm(_upper_triangle_iterate(len(names)), total=num_pairs, desc='Pairs'):
            # weights is (runs, output, input)
            runs_w_i = weights[:, i]
            runs_w_j = weights[:, j]
            metric_samples = np.array(list(metric_fn(w_i, w_j) for w_i, w_j in zip(runs_w_i, runs_w_j)))
            t_stat, p_value = ttest_1samp(metric_samples, population_mean)
            mu = np.mean(metric_samples).item()
            sigma = np.std(metric_samples, ddof=1).item()
            metric_statistics = PairwiseMetricStatistics(
                names[i], names[j], mu, sigma, t_stat, p_value, len(metric_samples))
            pickle.dump(metric_statistics, output_file, protocol=pickle.HIGHEST_PROTOCOL)


def _pairwise_metric_statistics(path):
    with open(path, 'rb') as f:
        num_items = pickle.load(f)
        for _ in range(num_items):
            yield pickle.load(f)


def _read_p_values(path):
    for metric_statistics in _pairwise_metric_statistics(path):
        yield metric_statistics.p_value


def filter_pairwise_metric_statistics(path, alpha=0.05, method='py'):
    for corrected_p_value, metric_statistics in zip(
            fdr_correction_file_based(_read_p_values(path), method=method), _pairwise_metric_statistics(path)):
        if corrected_p_value <= alpha:
            yield metric_statistics
