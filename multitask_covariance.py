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
    worker_device, worker_update_progress

replace_root_logger_handler()
logger = logging.getLogger(__name__)


def _io_setup(set_name, settings):
    hash_ = task_hash(settings)
    paths = Paths()
    paths.model_path = os.path.join(paths.model_path, set_name, hash_)
    paths.result_path = os.path.join(paths.result_path, set_name, hash_)

    corpus_dataset_factory = CorpusDatasetFactory(cache_path=paths.cache_path)

    return corpus_dataset_factory, paths


class RunningCovariance:

    def __init__(self, value=None):
        self.count = 0
        self.mu = None
        self.sum_sq_diff = None
        if value is not None:
            self.update(value)

    def update(self, value):
        if np.ndim(value) > 1:
            value = np.reshape(value, (-1, value.shape[-1]))
            for v in value:
                self.update(v)
            return

        if self.count == 0:
            self.mu = np.asarray(np.copy(value), dtype=np.float64)
            self.sum_sq_diff = np.zeros((len(value), len(value)), dtype=self.mu.dtype)

        if np.ndim(value) != 1 or len(value) != len(self.mu):
            raise ValueError('Shape must be 1d and consistent with previous updates')

        self.count += 1
        delta = value - self.mu
        self.mu += delta / self.count
        delta_2 = value - self.mu
        self.sum_sq_diff += np.expand_dims(delta, 1) * np.expand_dims(delta_2, 0)

    def mean(self):
        if self.count == 0:
            raise ValueError('No values')
        return self.mu

    def covariance(self, ddof=0):
        if self.count == 0:
            raise ValueError('No values')
        if self.count - ddof < 1:
            return np.full_like(self.sum_sq_diff, np.nan)
        return self.sum_sq_diff / (self.count - ddof)


def _worker_update_progress_wrap(count=1):
    worker_update_progress(count, ignore_if_no_progress_context=True)


def _worker_estimate_covariance_run(name, index_variation, index_run, covariance_names):
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

    _estimate_covariance_run(
        settings, paths, index_run, worker_device(), corpus_dataset_factory, progress_updater, covariance_names)


def _estimate_covariance_run(
        settings, paths, index_run, device, corpus_dataset_factory, progress_updater, covariance_names):

    output_dir = os.path.join(paths.result_path, 'run_{}'.format(index_run))

    completion_file_path = os.path.join(output_dir, 'completed.txt')

    if not os.path.exists(completion_file_path):
        raise ValueError('Path does not exist, run {} incomplete: {}'.format(index_run, completion_file_path))

    if os.path.exists(os.path.join(output_dir, 'covariance_estimates.npz')):
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

    if np.ndim(covariance_names) == 0:
        covariance_names = [covariance_names]

    result = dict()
    for batch in data_loader:
        # if len(all_results) % 1000 == 0:
        #     logger.info("Processing example: %d" % (len(all_results)))
        for k in batch:
            batch[k] = batch[k].to(device)
        batch['global_step'] = settings
        with torch.no_grad():
            predictions = model(batch, validation_data)
            for name in covariance_names:
                if name in predictions:
                    values = predictions[name].detach().cpu().numpy()
                    if np.ndim(values) > 2:
                        values = np.reshape(values, (-1, values.shape[-1]))
                    if 'response_id' in batch:
                        if not torch.all(predictions['response_id'] == predictions['response_id'][0]):
                            raise ValueError('Expected only one unique response_id per batch')
                        response_id = predictions['response_id'][0].item()
                        splits = ['__all__', validation_data.response_field_for_id(response_id)]
                    else:
                        splits = ['__all__']
                    for split in splits:
                        if split not in result:
                            result[split] = dict()
                        if name not in result[split]:
                            result[split][name] = RunningCovariance(values)
                        else:
                            result[split][name].update(values)

    result_dict = dict()
    active_names = set()
    for split in result:
        for name in result[split]:
            active_names.add(name)
            result_dict['mean_{}_{}'.format(name, split)] = result[split][name].mean()
            result_dict['covariance_{}_{}'.format(name, split)] = result[split][name].covariance()

    result_dict['splits'] = sorted(result)
    result_dict['names'] = sorted(active_names)

    np.savez(os.path.join(output_dir, 'covariance_estimates.npz'), **result_dict)

    progress_updater.update_run()
    return result


def estimate_covariance(
        set_name: str,
        settings: Settings,
        device: torch.device,
        covariance_names,
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
        _estimate_covariance_run(
            settings, paths, index_run, device, corpus_dataset_factory, progress_updater, covariance_names)

    if progress_bar is not None:
        progress_bar.close()


def main():
    parser = argparse.ArgumentParser(
        'Estimates the covariance of a layer in a trained model using the examples in the validation data')

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

    parser.add_argument('--covariances', action='store', required=False, default='',
                        help='Which features to compute covariances on. Can be a string or a comma separated list')

    args = parser.parse_args()

    logging.getLogger().setLevel(level=args.log_level.upper())

    named_settings = named_variations(args.name)

    covariances = [c.strip() for c in args.covariances.split(',') if len(c.strip()) > 0]
    if len(covariances) == 0:
        parser.error('Expected non-empty covariances')

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
                    func=_worker_estimate_covariance_run,
                    iterables=[
                        itertools.repeat(args.name, len(indices_variation)),
                        indices_variation,
                        indices_run,
                        itertools.repeat(covariances, len(indices_variation))],
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
                estimate_covariance(variation, settings_, device, covariances, progress_updater, index_run)

        progress_bar.close()


if __name__ == '__main__':
    main()
