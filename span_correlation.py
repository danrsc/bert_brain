from dataclasses import replace
import argparse
import logging
import os
import itertools

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader as TorchDataLoader
from tqdm import tqdm

from tqdm_logging import replace_root_logger_handler
from run_variations import ProgressUpdater
from bert_brain_paths import Paths
from bert_brain import named_variations, cuda_most_free_device, cuda_auto_empty_cache_context, task_hash, \
    CorpusDatasetFactory, Settings, set_random_seeds, DataIdMultiDataset, setup_prediction_heads_and_losses, \
    BertMultiPredictionHead, ReplaySampler, collate_fn, read_loss_curve, ProgressContext, cuda_map_unordered, \
    worker_device, worker_update_progress, KeyedSingleTargetSpanMaxPoolFactory

replace_root_logger_handler()
logger = logging.getLogger(__name__)


def _io_setup(set_name, settings):
    hash_ = task_hash(settings)
    paths = Paths()
    paths.model_path = os.path.join(paths.model_path, set_name, hash_)
    paths.result_path = os.path.join(paths.result_path, set_name, hash_)

    corpus_dataset_factory = CorpusDatasetFactory(cache_path=paths.cache_path)

    return corpus_dataset_factory, paths


class RunningVariance:

    def __init__(self, value=None):
        self.count = 0
        self.mu = None
        self.sum_sq_diff = None
        if value is not None:
            self.update(value)

    def update(self, value):
        if np.ndim(value) > 1:
            raise ValueError('Expected a scalar, or a batch of scalars')
        if np.ndim(value) == 1:
            for v in value:
                self.update(v)
            return

        if self.count == 0:
            self.mu = np.asarray(np.copy(value), dtype=np.float64)
            self.sum_sq_diff = np.zeros_like(self.mu)

        self.count += 1
        delta = value - self.mu
        self.mu += delta / self.count
        delta_2 = value - self.mu
        self.sum_sq_diff += delta * delta_2

    def mean(self):
        if self.count == 0:
            raise ValueError('No values')
        return self.mu

    def variance(self, ddof=0):
        if self.count == 0:
            raise ValueError('No values')
        if self.count - ddof < 1:
            return np.full_like(self.sum_sq_diff, np.nan)
        return self.sum_sq_diff / (self.count - ddof)


def _worker_update_progress_wrap(count=1):
    worker_update_progress(count, ignore_if_no_progress_context=True)


def _worker_span_correlation_run(name, index_variation, index_run, sequence_output_name, hdr_output_name):
    named_settings = named_variations(name)
    settings = None
    variation = None
    for index_current_variation, (variation_, training_variation) in enumerate(named_settings):
        if index_current_variation == index_variation:
            settings = named_settings[(variation_, training_variation)]
            variation = variation_
            break
    if settings is None:
        raise IndexError('No variation found for index_variation {} in name: {}'.format(index_variation, name))

    corpus_dataset_factory, paths = _io_setup(variation, settings)

    progress_updater = ProgressUpdater(update_run=_worker_update_progress_wrap)

    _estimate_span_correlation_run(
        settings, paths, index_run, worker_device(), corpus_dataset_factory, progress_updater, sequence_output_name,
        hdr_output_name)


def _estimate_span_correlation_run(
        settings, paths, index_run, device, corpus_dataset_factory, progress_updater,
        sequence_output_name, hdr_output_name):

    output_dir = os.path.join(paths.result_path, 'run_{}'.format(index_run))

    completion_file_path = os.path.join(output_dir, 'completed.txt')

    if not os.path.exists(completion_file_path):
        raise ValueError('Path does not exist, run {} incomplete: {}'.format(index_run, completion_file_path))

    if os.path.exists(os.path.join(output_dir, 'span_correlation.npz')):
        progress_updater.update_run()
        return

    output_model_path = os.path.join(paths.model_path, 'run_{}'.format(index_run))

    # hack to get the global step
    loss_curves = read_loss_curve(os.path.join(output_dir, 'validation_curve.npz'))
    global_step = 0
    for k in loss_curves:
        global_step = max(global_step, np.max(loss_curves[k][1]))

    set_random_seeds(settings.seed, index_run, n_gpu=1)

    data_set_paths = list()
    for corpus in settings.corpora:
        data_set_paths.append(corpus_dataset_factory.maybe_make_data_set_files(
            index_run,
            corpus,
            settings.preprocessors,
            settings.get_split_function(corpus, index_run),
            settings.preprocess_fork_fn,
            False,
            paths,
            settings.max_sequence_length,
            settings.create_meta_train_dataset))

    train_data, validation_data, test_data, meta_train_data = (
        DataIdMultiDataset(
            which,
            data_set_paths,
            settings.all_loss_tasks,
            data_id_in_batch_keys=settings.data_id_in_batch_keys,
            filter_when_not_in_loss_keys=settings.filter_when_not_in_loss_keys,
            field_spec_replacers=settings.field_spec_replacers)
        for which in ('train', 'validation', 'test', 'meta_train'))

    for key in settings.head_graph_parts:
        for inner_key in settings.head_graph_parts[key]:
            if isinstance(settings.head_graph_parts[key][inner_key], KeyedSingleTargetSpanMaxPoolFactory):
                settings.head_graph_parts[key][inner_key] = replace(
                    settings.head_graph_parts[key][inner_key], output_span_representations=True)

    graph_parts, common_graph_keys, token_supplemental_key_to_shape, pooled_supplemental_key_to_shape, loss_handlers = \
        setup_prediction_heads_and_losses(settings, train_data)

    # Prepare model
    model = BertMultiPredictionHead.from_pretrained(
        output_model_path,
        head_graph_parts=graph_parts,
        token_supplemental_key_to_shape=token_supplemental_key_to_shape,
        pooled_supplemental_key_to_shape=pooled_supplemental_key_to_shape)
    if settings.optimization_settings.fp16:
        model.half()
    model.to(device)

    model.eval()  # should already be eval, but just make sure

    batch_sampler = ReplaySampler(
        validation_data,
        os.path.join(output_dir, 'validation_predictions'),
        settings.optimization_settings.predict_batch_size)
    data_loader = TorchDataLoader(
        validation_data, batch_sampler=batch_sampler, collate_fn=collate_fn,
        num_workers=settings.optimization_settings.num_loader_workers)

    result = dict()
    for batch in data_loader:
        for k in batch:
            batch[k] = batch[k].to(device)
        batch['global_step'] = settings
        with torch.no_grad():
            predictions = model(batch, validation_data)

            assert('response_id' in predictions)
            if not torch.all(predictions['response_id'] == predictions['response_id'][0]):
                raise ValueError('Expected only one unique response_id per batch')
            response_id = predictions['response_id'][0].item()
            response_field = validation_data.response_field_for_id(response_id)

            # (batch, sequence, channel)
            sequence_output = predictions[sequence_output_name].detach().cpu().numpy()
            index_span = 0
            span_outputs = list()
            while True:
                span_output_name = sequence_output_name + '_span{}'.format(index_span)
                if span_output_name not in predictions:
                    break
                span_outputs.append(predictions[span_output_name].detach().cpu().numpy())
                index_span += 1

            hdr_output = predictions[hdr_output_name].detach().cpu().numpy()

            for index_batch_item, sequence in enumerate(sequence_output):
                prefix = response_field
                is_classification_loss = any(
                    h.field == response_field and h.is_classification_loss() for h in loss_handlers)
                if is_classification_loss:
                    label = int(batch[response_field][index_batch_item].item())
                    text_labels = validation_data.text_labels_for_field(response_field)
                    if text_labels is not None:
                        label = text_labels[label]
                    prefix = '{}_{}'.format(prefix, label)

                # (sequence, channel)
                indicator_valid = np.all(np.isfinite(sequence), axis=-1)
                # make sure all the valid tokens are at the start of the sequence
                assert(np.sum(indicator_valid) == np.sum(indicator_valid[:np.sum(indicator_valid)]))
                sequence = sequence[:np.sum(indicator_valid)]
                cls = sequence[0]
                sep = sequence[-1]
                sequence = sequence[1:-1]

                cls_sep_key = '{}_cls_sep'.format(prefix)
                if cls_sep_key not in result:
                    result[cls_sep_key] = RunningVariance()
                result[cls_sep_key].update(np.corrcoef(cls, sep)[0, 1])

                for index_token, token in enumerate(sequence):
                    cls_key = '{}_cls_{}'.format(prefix, index_token)
                    if cls_key not in result:
                        result[cls_key] = RunningVariance()
                    result[cls_key].update(np.corrcoef(cls, token)[0, 1])
                    sep_key = '{}_sep_{}'.format(prefix, index_token)
                    if sep_key not in result:
                        result[sep_key] = RunningVariance()
                    result[sep_key].update(np.corrcoef(sep, token)[0, 1])
                    # for index_token_2 in range(index_token + 1, len(sequence)):
                    #     key = '{}_{}_{}'.format(prefix, index_token, index_token_2)
                    #     if key not in result:
                    #         result[key] = RunningVariance()
                    #     result[key].update(np.corrcoef(token, sequence[index_token_2])[0, 1])
                    for relative in range(-3, 4):
                        if len(sequence) > index_token + relative >= 0:
                            key = '{}_rel_{}'.format(prefix, relative)
                            if key not in result:
                                result[key] = RunningVariance()
                            result[key].update(np.corrcoef(token, sequence[relative + index_token])[0, 1])

                if len(span_outputs) > 0:
                    for index_span, span in enumerate(span_outputs):
                        cls_key = '{}_cls_span_{}'.format(prefix, index_span)
                        if cls_key not in result:
                            result[cls_key] = RunningVariance()
                        result[cls_key].update(np.corrcoef(cls, span[index_batch_item])[0, 1])
                        sep_key = '{}_sep_span_{}'.format(prefix, index_span)
                        if sep_key not in result:
                            result[sep_key] = RunningVariance()
                        result[sep_key].update(np.corrcoef(sep, span[index_batch_item])[0, 1])
                        for index_token in range(len(sequence)):
                            key = '{}_{}_span_{}'.format(prefix, index_token, index_span)
                            if key not in result:
                                result[key] = RunningVariance()
                            result[key].update(np.corrcoef(span[index_batch_item], sequence[index_token])[0, 1])
                        for index_span_2 in range(index_span + 1, len(span_outputs)):
                            key = '{}_span_{}_span_{}'.format(prefix, index_span, index_span_2)
                            if key not in result:
                                result[key] = RunningVariance()
                            result[key].update(np.corrcoef(span, span_outputs[index_span_2][index_batch_item])[0, 1])

                cls_key = '{}_cls_hdr'.format(prefix)
                if cls_key not in result:
                    result[cls_key] = RunningVariance()
                result[cls_key].update(np.corrcoef(cls, hdr_output[index_batch_item])[0, 1])
                sep_key = '{}_sep_hdr'.format(prefix)
                if sep_key not in result:
                    result[sep_key] = RunningVariance()
                result[sep_key].update(np.corrcoef(sep, hdr_output[index_batch_item])[0, 1])
                for index_token in range(len(sequence)):
                    key = '{}_{}_hdr'.format(prefix, index_token)
                    if key not in result:
                        result[key] = RunningVariance()
                    result[key].update(np.corrcoef(hdr_output[index_batch_item], sequence[index_token])[0, 1])

    result_dict = dict()
    for key in result:
        result_dict['mean_{}'.format(key)] = result[key].mean()
        result_dict['variance_{}'.format(key)] = result[key].variance()

    result_dict['names'] = sorted(result)

    np.savez(os.path.join(output_dir, 'span_correlation.npz'), **result_dict)

    progress_updater.update_run()
    return result


def estimate_span_correlation(
        set_name: str,
        settings: Settings,
        device: torch.device,
        sequence_output_name,
        hdr_output_name,
        progress_updater=None,
        index_run=None):

    corpus_dataset_factory, paths = _io_setup(set_name, settings)

    progress_bar = None
    if progress_updater is None:
        progress_bar = tqdm(total=settings.num_runs, desc='Runs')
        progress_updater = ProgressUpdater(update_run=progress_bar.update)

    if index_run is not None:
        if np.ndim(index_run) == 0:
            runs = [index_run]
        else:
            runs = index_run
    else:
        runs = range(settings.num_runs)

    for index_run in runs:
        _estimate_span_correlation_run(
            settings, paths, index_run, device, corpus_dataset_factory, progress_updater, sequence_output_name,
            hdr_output_name)

    if progress_bar is not None:
        progress_bar.close()


def main():
    parser = argparse.ArgumentParser(
        'Estimates the correlation between tokens, spans, etc. of a layer in a trained model using the examples '
        'in the validation data')

    parser.add_argument('--log_level', action='store', required=False, default='WARNING',
                        help='Sets the log-level. Defaults to WARNING')

    parser.add_argument('--index_run', action='store', required=False, default=-1, type=int,
                        help='Specify to train a particular run. Mostly useful for debugging')

    parser.add_argument('--num_workers', action='store', required=False, default=1, type=int,
                        help='If 0, model will be run on CPU, If > 1, runs will be executed in parallel on GPUs')

    parser.add_argument('--min_memory_gb', action='store', required=False, default=4, type=int,
                        help='How many GB must be free on a GPU for a worker to run on that GPU. '
                             'Ignored if num_workers < 2')

    parser.add_argument(
        '--name', action='store', required=False, default='', help='Which set to run')

    parser.add_argument('--sequence_output_name', action='store', required=False, default='sequence_all_bottleneck',
                        help='The sequence output name we are computing the correlations over')
    parser.add_argument('--hdr_output_name', action='store', required=False, default='hdr_pooled',
                        help='The hemodynamic response pooled output name')

    args = parser.parse_args()

    logging.getLogger().setLevel(level=args.log_level.upper())

    named_settings = named_variations(args.name)

    if args.num_workers > 1:
        indices_variation = list()
        indices_run = list()
        for index_variation, k in enumerate(named_settings):
            indices_variation.extend([index_variation] * named_settings[k].num_runs)
            indices_run.extend(range(named_settings[k].num_runs))
        with ProgressContext(
                total=sum(named_settings[k].num_runs for k in named_settings), desc='Runs') as progress_context:
            for _ in cuda_map_unordered(
                    min_memory_gb=args.min_memory_gb,
                    func=_worker_span_correlation_run,
                    iterables=[
                        itertools.repeat(args.name, len(indices_variation)),
                        indices_variation,
                        indices_run,
                        itertools.repeat(args.sequence_output_name, len(indices_variation)),
                        itertools.repeat(args.hdr_output_name, len(indices_variation))],
                    max_workers=args.num_workers,
                    mp_context=progress_context):
                pass
    else:
        progress_bar = tqdm(total=sum(named_settings[k].num_runs for k in named_settings), desc='Runs')
        progress_updater = ProgressUpdater(update_run=progress_bar.update)

        for variation, training_variation in named_settings:
            settings_ = named_settings[(variation, training_variation)]
            if args.num_workers == 0:
                settings_ = replace(settings_, no_cuda=True)

            if not torch.cuda.is_available or settings_.no_cuda:
                device = torch.device('cpu')
            else:
                device_id, free = cuda_most_free_device()
                device = torch.device('cuda', device_id)
                logger.info('binding to device {} with {} memory free'.format(device_id, free))
            logger.info("device: {}, 16-bits trainiing: {}".format(
                device,
                settings_.optimization_settings.fp16))

            with cuda_auto_empty_cache_context(device):
                index_run = None if args.index_run < 0 else args.index_run
                estimate_span_correlation(variation, settings_, device, args.sequence_output_name, args.hdr_output_name,
                                          progress_updater, index_run)

        progress_bar.close()


if __name__ == '__main__':
    main()
