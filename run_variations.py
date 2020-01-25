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
from shutil import rmtree
import argparse
import logging
import os
from dataclasses import replace
from typing import Optional
from tqdm import tqdm
from tqdm_logging import replace_root_logger_handler

import torch

from bert_brain import cuda_most_free_device, cuda_auto_empty_cache_context, CorpusDatasetFactory, \
    Settings, task_hash, set_random_seeds, named_variations, singleton_variation, train, DataIdMultiDataset
from bert_brain_paths import Paths


__all__ = ['run_variation']


replace_root_logger_handler()
logger = logging.getLogger(__name__)


def progress_iterate(iterable, progress_bar):
    for item in iterable:
        yield item
        progress_bar.update()


def run_variation(
            set_name: str,
            settings: Settings,
            force_cache_miss_set: Optional[set],
            device: torch.device,
            n_gpu: int,
            progress_bar=None):

    if settings.optimization_settings.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            settings.optimization_settings.gradient_accumulation_steps))

    # TODO: seems like this is taken care of below?
    # settings = replace(
    #   settings, train_batch_size=int(settings.train_batch_size / settings.gradient_accumulation_steps))

    def io_setup():
        hash_ = task_hash(settings)
        paths_ = Paths()
        paths_.model_path = os.path.join(paths_.model_path, set_name, hash_)
        paths_.result_path = os.path.join(paths_.result_path, set_name, hash_)

        corpus_dataset_factory_ = CorpusDatasetFactory(cache_path=paths_.cache_path)

        if not os.path.exists(paths_.model_path):
            os.makedirs(paths_.model_path)
        if not os.path.exists(paths_.result_path):
            os.makedirs(paths_.result_path)

        return corpus_dataset_factory_, paths_

    corpus_dataset_factory, paths = io_setup()

    if progress_bar is None:
        progress_bar = tqdm(total=settings.num_runs, desc='Runs')

    for index_run in progress_iterate(range(settings.num_runs), progress_bar):

        output_dir = os.path.join(paths.result_path, 'run_{}'.format(index_run))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        completion_file_path = os.path.join(output_dir, 'completed.txt')

        if os.path.exists(completion_file_path):
            continue

        output_model_path = os.path.join(paths.model_path, 'run_{}'.format(index_run))

        seed = set_random_seeds(settings.seed, index_run, n_gpu)

        data_set_paths = list()
        for corpus in settings.corpora:
            data_set_paths.append(corpus_dataset_factory.maybe_make_data_set_files(
                seed,
                index_run,
                corpus,
                settings.preprocessors,
                output_model_path,
                settings.get_split_function(corpus.corpus_key, index_run),
                settings.preprocess_fork_fn,
                force_cache_miss_set is not None and (
                        corpus.corpus_key in force_cache_miss_set or '__all__' in force_cache_miss_set),
                paths,
                settings.max_sequence_length))

        train_data, validation_data, test_data = (
            DataIdMultiDataset(
                which,
                data_set_paths,
                settings.all_loss_tasks,
                data_id_in_batch_keys=settings.data_id_in_batch_keys,
                filter_when_not_in_loss_keys=settings.filter_when_not_in_loss_keys)
            for which in ('train', 'validation', 'test'))

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

        train(settings, output_dir, completion_file_path, output_model_path,
              train_data, validation_data, test_data, n_gpu, device, load_from_path)


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

    parser.add_argument('--no_cuda', action='store_true', required=False, help='If specified, model will be run on CPU')

    parser.add_argument(
        '--name', action='store', required=False, default='erp', help='Which set to run')

    args = parser.parse_args()

    force_cache_miss_set = None
    if args.force_cache_miss is not None:
        force_cache_miss_set = set(k for k in args.force_cache_miss.split(','))

    logging.getLogger().setLevel(level=args.log_level.upper())

    if args.clean:
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

    named_settings = named_variations(args.name)
    progress_bar = tqdm(total=sum(named_settings[k].num_runs for k in named_settings), desc='Runs')
    for variation, training_variation in named_settings:
        settings_ = named_settings[(variation, training_variation)]
        if args.no_cuda:
            settings_ = replace(settings_, no_cuda=True)

        if settings_.optimization_settings.local_rank == -1 or settings_.no_cuda:
            if not torch.cuda.is_available or settings_.no_cuda:
                device = torch.device('cpu')
            else:
                device_id, free = cuda_most_free_device()
                device = torch.device('cuda', device_id)
                logger.info('binding to device {} with {} memory free'.format(device_id, free))
            n_gpu = 1  # torch.cuda.device_count()
        else:
            device = torch.device('cuda', settings_.local_rank)
            n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            # noinspection PyUnresolvedReferences
            torch.distributed.init_process_group(backend='nccl')
            if settings_.optimization_settings.fp16:
                logger.info("16-bits training currently not supported in distributed training")
                settings_.optimization_settings.fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)
        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits trainiing: {}".format(
            device,
            n_gpu,
            bool(settings_.optimization_settings.local_rank != -1),
            settings_.optimization_settings.fp16))

        with cuda_auto_empty_cache_context(device):
            run_variation(variation, settings_, force_cache_miss_set, device, n_gpu, progress_bar=progress_bar)

    progress_bar.close()


if __name__ == '__main__':
    main()
