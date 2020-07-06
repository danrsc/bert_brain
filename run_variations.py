# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import shutil
from shutil import rmtree
import argparse
import logging
import os
import itertools
from dataclasses import replace, dataclass
from typing import Optional, Callable
from tqdm import tqdm
from tqdm_logging import replace_root_logger_handler
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

from bert_brain import cuda_most_free_device, cuda_auto_empty_cache_context, ProgressContext, cuda_map_unordered, \
    worker_update_progress, worker_update_progress_total, worker_device, CorpusDatasetFactory, Settings, task_hash, \
    set_random_seeds, named_variations, singleton_variation, train, DataIdMultiDataset
from bert_brain_paths import Paths


__all__ = ['run_variation', 'ProgressUpdater']


replace_root_logger_handler()
logger = logging.getLogger(__name__)


def _dummy_update(_=1):
    return


def _dummy_update_total(_):
    return


def _worker_update_progress_wrap(count=1):
    worker_update_progress(count, ignore_if_no_progress_context=True)


def _worker_update_progress_total_wrap(increment):
    worker_update_progress_total(increment, ignore_if_no_progress_context=True)


@dataclass(frozen=True)
class ProgressUpdater:
    update_batch: Callable[[int], None] = _dummy_update
    update_epoch: Callable[[int], None] = _dummy_update
    update_run: Callable[[int], None] = _dummy_update
    update_batch_total: Callable[[int], None] = _dummy_update_total


def _worker_run_variation(name, index_variation, index_run, force_cache_miss_set, progress_level):
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

    progress_updater_kwargs = dict()
    if progress_level == 'runs':
        progress_updater_kwargs['update_run'] = _worker_update_progress_wrap
    elif progress_level == 'epochs':
        progress_updater_kwargs['update_epoch'] = _worker_update_progress_wrap
    elif progress_level == 'batches':
        progress_updater_kwargs['update_batch'] = _worker_update_progress_wrap
        progress_updater_kwargs['update_batch_total'] = _worker_update_progress_total_wrap
    else:
        raise ValueError('Unknown progress_level: {}'.format(progress_level))
    progress_updater = ProgressUpdater(**progress_updater_kwargs)

    _train_single_run(
        settings, paths, index_run, worker_device(), corpus_dataset_factory, force_cache_miss_set, progress_updater)


def _io_setup(set_name, settings):
    hash_ = task_hash(settings)
    paths = Paths()
    paths.model_path = os.path.join(paths.model_path, set_name, hash_)
    paths.result_path = os.path.join(paths.result_path, set_name, hash_)

    corpus_dataset_factory = CorpusDatasetFactory(cache_path=paths.cache_path)

    if not os.path.exists(paths.model_path):
        os.makedirs(paths.model_path)
    if not os.path.exists(paths.result_path):
        os.makedirs(paths.result_path)

    return corpus_dataset_factory, paths


def _train_single_run(
        settings, paths, index_run, device, corpus_dataset_factory, force_cache_miss_set, progress_updater):
    output_dir = os.path.join(paths.result_path, 'run_{}'.format(index_run))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    completion_file_path = os.path.join(output_dir, 'completed.txt')

    if os.path.exists(completion_file_path):
        progress_updater.update_run()
        progress_updater.update_epoch(settings.optimization_settings.num_train_epochs)
        with open(completion_file_path, 'rt') as completion_file:
            for line in completion_file:
                completion_info = line.strip().split('\t')
                if len(completion_info) == 2 and completion_info[0] == 'batches':
                    num_batches = int(completion_info[1])
                    progress_updater.update_batch_total(num_batches)
                    progress_updater.update_batch(num_batches)
        return

    output_model_path = os.path.join(paths.model_path, 'run_{}'.format(index_run))

    set_random_seeds(settings.seed, index_run, n_gpu=1)

    data_set_paths = list()
    for corpus in settings.corpora:
        data_set_paths.append(corpus_dataset_factory.maybe_make_data_set_files(
            index_run,
            corpus,
            settings.preprocessors,
            settings.get_split_function(corpus, index_run),
            settings.preprocess_fork_fn,
            force_cache_miss_set is not None and (
                    corpus.corpus_key in force_cache_miss_set or '__all__' in force_cache_miss_set),
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

    load_from_path = None
    if settings.load_from is not None:
        # noinspection PyCallingNonCallable
        (load_from_variation, _), load_from_settings = singleton_variation(settings.load_from)
        # noinspection PyCallingNonCallable
        load_from_index_run = index_run \
            if settings.load_from_run_map is None else settings.load_from_run_map(index_run)
        load_from_path = os.path.join(
            Paths().model_path,
            load_from_variation,
            task_hash(load_from_settings),
            'run_{}'.format(load_from_index_run))

    train(replace(settings), output_dir, completion_file_path, output_model_path,
          train_data, validation_data, test_data, meta_train_data, device, progress_updater, load_from_path)
    progress_updater.update_run()


def _reachable_for_variation(variation_name, training_variation):
    data_set_paths = set()
    model_paths = set()
    result_paths = set()

    named_settings = named_variations(variation_name)
    settings = named_settings[variation_name, training_variation]

    corpus_dataset_factory, paths = _io_setup(variation_name, settings)
    for index_run in range(settings.num_runs):
        model_paths.add(os.path.join(paths.model_path, 'run_{}'.format(index_run)))
        result_paths.add(os.path.join(paths.result_path, 'run_{}'.format(index_run)))
        set_random_seeds(settings.seed, index_run, n_gpu=1)

        for corpus in settings.corpora:
            data_set_path = corpus_dataset_factory.maybe_make_data_set_files(
                index_run,
                corpus,
                settings.preprocessors,
                settings.get_split_function(corpus, index_run),
                settings.preprocess_fork_fn,
                False,
                paths,
                settings.max_sequence_length,
                settings.create_meta_train_dataset,
                paths_only=True)
            if data_set_path is not None and len(data_set_path) > 0:
                data_set_paths.add(data_set_path)

    return data_set_paths, model_paths, result_paths


def find_reachable():
    full_variation_map = named_variations('__full_map__')
    data_set_paths = set()
    model_paths = set()
    result_paths = set()

    with ProcessPoolExecutor() as ex:
        futures = list(
            ex.submit(_reachable_for_variation, v, t) for v, t in full_variation_map)
        for future in tqdm(as_completed(futures), total=len(futures), desc='variations'):
            d, m, r = future.result()
            data_set_paths.update(d)
            model_paths.update(m)
            result_paths.update(r)

    return data_set_paths, model_paths, result_paths


def _reachable_closure(reachable):
    closure = set()
    for entry in reachable:
        while True:
            up_one, _ = os.path.split(entry)
            if len(up_one) == 0 or up_one == '/':
                break
            closure.add(up_one)
            entry = up_one
    reachable = set(reachable)
    reachable.update(closure)
    return reachable


def _leaf_count_at_least(base_path, remaining_depth, count):
    if count is None:
        return False
    if count == 'force':
        return True
    leaf_count = 0
    with os.scandir(base_path) as it:
        for entry in it:
            if not entry.is_dir():
                continue
            if remaining_depth == 0:
                leaf_count += 1
                if leaf_count > count:
                    return True
            elif _leaf_count_at_least(os.path.join(base_path, entry.name), remaining_depth - 1, count):
                return True
    return False


def _find_unreachable(base_path, remaining_depth, reachable, leave_if_leaf_count_at_least=None):
    unreachable = list()
    count_reachable = 0
    with os.scandir(base_path) as it:
        for entry in it:
            if not entry.is_dir():
                continue
            entry_path = os.path.join(base_path, entry.name)
            leave_if = leave_if_leaf_count_at_least
            if _leaf_count_at_least(entry_path, remaining_depth - 1, leave_if):
                leave_if = 'force'
            if entry_path not in reachable and leave_if != 'force':
                unreachable.append(entry_path)
            elif remaining_depth > 0:
                entry_unreachable, entry_count_reachable = _find_unreachable(
                    entry_path, remaining_depth - 1, reachable, leave_if)
                unreachable.extend(entry_unreachable)
                count_reachable += entry_count_reachable
            else:
                count_reachable += 1
    return unreachable, count_reachable


def find_unreachable_data_set_paths(reachable):
    unreachable, count_reachable = _find_unreachable(Paths().cache_path, 1, _reachable_closure(reachable))
    print('Found {} reachable and {} unreachable data set directories'.format(count_reachable, len(unreachable)))
    return unreachable


def find_unreachable_model_paths(reachable, leave_if_leaf_count_at_least=None):
    unreachable, count_reachable = _find_unreachable(
        Paths().model_path, 2, _reachable_closure(reachable),
        leave_if_leaf_count_at_least=leave_if_leaf_count_at_least)
    print('Found {} reachable and {} unreachable model directories'.format(count_reachable, len(unreachable)))
    return unreachable


def find_unreachable_result_paths(reachable, leave_if_leaf_count_at_least=None):
    unreachable, count_reachable = _find_unreachable(
        Paths().result_path, 2, _reachable_closure(reachable),
        leave_if_leaf_count_at_least=leave_if_leaf_count_at_least)
    print('Found {} reachable and {} unreachable result directories'.format(count_reachable, len(unreachable)))
    return unreachable


def run_variation(
            set_name: str,
            settings: Settings,
            force_cache_miss_set: Optional[set],
            device: torch.device,
            progress_updater=None,
            index_run=None):

    if settings.optimization_settings.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            settings.optimization_settings.gradient_accumulation_steps))

    corpus_dataset_factory, paths = _io_setup(set_name, settings)

    progress_bar = None
    if progress_updater is None:
        progress_bar = tqdm(total=settings.num_runs, desc='Runs')
        progress_updater = ProgressUpdater(update_run=progress_bar.update)

    if index_run is not None:
        runs = [index_run]
    else:
        runs = range(settings.num_runs)
    for index_run in runs:
        _train_single_run(
            settings, paths, index_run, device, corpus_dataset_factory, force_cache_miss_set, progress_updater)

    if progress_bar is not None:
        progress_bar.close()


def _update_batch_total(progress_bar, increment):
    if progress_bar.total is None:
        progress_bar.total = increment
    else:
        progress_bar.total = progress_bar.total + increment


def main():
    parser = argparse.ArgumentParser(
        'Trains a regression by fine-tuning on specified task-variations starting with a BERT pretrained model')

    parser.add_argument('--clean', action='store_true', required=False,
                        help='DANGER: If specified, the current results will'
                             ' be removed and we will start from scratch')

    parser.add_argument('--force_cache_miss', action='store', required=False,
                        help='Data from the specified corpus will be loaded from raw files and then recached. '
                             'Useful if loading logic has changed')

    parser.add_argument('--log_level', action='store', required=False, default='WARNING',
                        help='Sets the log-level. Defaults to WARNING')

    parser.add_argument('--progress_level', action='store', required=False, default='runs',
                        help='Sets the resolution of the progress bar. One of (runs, epochs, batches)')

    parser.add_argument('--num_workers', action='store', required=False, default=1, type=int,
                        help='If 0, model will be run on CPU, If > 1, runs will be executed in parallel on GPUs')

    parser.add_argument('--min_memory_gb', action='store', required=False, default=4, type=int,
                        help='How many GB must be free on a GPU for a worker to run on that GPU. '
                             'Ignored if num_workers < 2')

    parser.add_argument('--index_run', action='store', required=False, default=-1, type=int,
                        help='Specify to train a particular run. Mostly useful for debugging')

    parser.add_argument('--archive_unreachable', action='store', required=False, default='',
                        help='Goes through all current experiments, computing the set of cache and model paths that '
                             'they comprise, and outputs the set of cache and model paths which is not reachable. '
                             'These can be used to delete unreachable data through rsync for example.')

    parser.add_argument(
        '--name', action='store', required=False, default='', help='Which set to run')

    args = parser.parse_args()

    force_cache_miss_set = None
    if args.force_cache_miss is not None:
        force_cache_miss_set = set(k for k in args.force_cache_miss.split(','))

    logging.getLogger().setLevel(level=args.log_level.upper())

    if args.clean:
        if len(args.name) == 0:
            sys.exit(0)

        while True:
            answer = input('About to remove results at {}. Is this really what you want to do [y/n]? '.format(
                args.name))
            if answer in {'Y', 'y', 'N', 'n'}:
                if answer == 'N' or answer == 'n':
                    print('No action taken')
                    sys.exit(0)
                break

        paths_ = Paths()

        variation_name = args.name
        try:
            named_settings = named_variations(args.name)
            for k in named_settings:
                variation_name = k[0]
                break
        except (TypeError, KeyError, ValueError):
            pass

        model_path = os.path.join(paths_.model_path, variation_name)
        result_path = os.path.join(paths_.result_path, variation_name)
        if os.path.exists(model_path):
            rmtree(model_path)
        if os.path.exists(result_path):
            rmtree(result_path)
        sys.exit(0)

    if len(args.archive_unreachable) > 0:
        data_set_paths, model_paths, result_paths = find_reachable()
        for path in find_unreachable_data_set_paths(data_set_paths):
            new_path = os.path.join(args.archive_unreachable, path[1:])
            print(new_path)
            os.makedirs(new_path, exist_ok=True)
            shutil.move(path, new_path)
        for path in find_unreachable_model_paths(model_paths, leave_if_leaf_count_at_least=10):
            new_path = os.path.join(args.archive_unreachable, path[1:])
            print(new_path)
            os.makedirs(new_path, exist_ok=True)
            shutil.move(path, new_path)
        for path in find_unreachable_result_paths(result_paths, leave_if_leaf_count_at_least=10):
            new_path = os.path.join(args.archive_unreachable, path[1:])
            print(new_path)
            os.makedirs(new_path, exist_ok=True)
            shutil.move(path, new_path)
        sys.exit(0)

    named_settings = named_variations(args.name)

    total = None
    if args.progress_level == 'runs':
        total = sum(named_settings[k].num_runs for k in named_settings)
        desc = 'Runs'
    elif args.progress_level == 'epochs':
        total = sum(named_settings[k].num_runs * named_settings[k].optimization_settings.num_train_epochs
                    for k in named_settings)
        desc = 'Epochs'
    elif args.progress_level == 'batches':
        # leave total unspecified here, we will dynamically compute it
        desc = 'Batches'
    else:
        raise ValueError('Unknown value for progress_level: {}'.format(args.progress_level))

    if args.num_workers > 1:
        indices_variation = list()
        indices_run = list()
        for index_variation, k in enumerate(named_settings):
            indices_variation.extend([index_variation] * named_settings[k].num_runs)
            indices_run.extend(range(named_settings[k].num_runs))
        with ProgressContext(total=total, desc=desc) as progress_context:
            for _ in cuda_map_unordered(
                    min_memory_gb=args.min_memory_gb,
                    func=_worker_run_variation,
                    iterables=[
                        itertools.repeat(args.name, len(indices_variation)),
                        indices_variation,
                        indices_run,
                        itertools.repeat(force_cache_miss_set, len(indices_variation)),
                        itertools.repeat(args.progress_level)],
                    max_workers=args.num_workers,
                    mp_context=progress_context):
                pass
    else:
        progress_bar = tqdm(total=total, desc=desc)

        progress_updater_kwargs = dict()
        if args.progress_level == 'runs':
            progress_updater_kwargs['update_run'] = progress_bar.update
        elif args.progress_level == 'epochs':
            progress_updater_kwargs['update_epoch'] = progress_bar.update
        elif args.progress_level == 'batches':
            progress_updater_kwargs['update_batch'] = progress_bar.update
            progress_updater_kwargs['update_batch_total'] = lambda increment: _update_batch_total(
                progress_bar, increment)
        else:
            raise ValueError('Unknown progress_level: {}'.format(args.progress_level))
        progress_updater = ProgressUpdater(**progress_updater_kwargs)

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
                run_variation(variation, settings_, force_cache_miss_set, device, progress_updater, index_run)

        progress_bar.close()


if __name__ == '__main__':
    main()
