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
import itertools
import logging
import os
import random
from dataclasses import replace
from typing import Sequence
import hashlib
from tqdm import trange
from tqdm_logging import replace_root_logger_handler

import numpy as np
import torch

from bert_erp_common import SwitchRemember
from bert_erp_datasets import DataKeys, DataPreparer
from bert_erp_settings import Settings
from bert_erp_paths import Paths
from train_eval import train, make_datasets

__all__ = ['task_hash', 'named_variations', 'run_variation', 'iterate_powerset']


replace_root_logger_handler()
logger = logging.getLogger(__name__)


def task_hash(loss_tasks):
    hash_ = hashlib.sha256()
    for loss_task in sorted(loss_tasks):
        hash_.update(loss_task.encode())
    return hash_.hexdigest()


def _seed(seed, index_run, n_gpu):
    hash_ = hashlib.sha256('{}'.format(seed).encode())
    hash_.update('{}'.format(index_run).encode())
    seed = np.frombuffer(hash_.digest(), dtype='uint32')
    random_state = np.random.RandomState(seed)
    np.random.set_state(random_state.get_state())
    seed = np.random.randint(low=0, high=np.iinfo('uint32').max)
    random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
    return seed


def run_variation(
            set_name,
            loss_tasks: Sequence[str],
            settings: Settings,
            num_runs: int,
            auxiliary_loss_tasks: Sequence[str],
            force_cache_miss: bool):

    if settings.local_rank == -1 or settings.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not settings.no_cuda else "cpu")
        n_gpu = 1  # torch.cuda.device_count()
    else:
        device = torch.device("cuda", settings.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if settings.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            settings.fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits trainiing: {}".format(
        device, n_gpu, bool(settings.local_rank != -1), settings.fp16))

    if settings.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            settings.gradient_accumulation_steps))

    # TODO: seems like this is taken care of below?
    # settings = replace(
    #   settings, train_batch_size=int(settings.train_batch_size / settings.gradient_accumulation_steps))

    def io_setup():
        temp_paths = Paths()
        data_loader_ = temp_paths.make_data_loader(data_key_kwarg_dict=settings.data_key_kwargs)
        hash_ = task_hash(loss_tasks)
        model_path_ = os.path.join(temp_paths.model_path, set_name, hash_)
        result_path_ = os.path.join(temp_paths.result_path, set_name, hash_)

        if not os.path.exists(model_path_):
            os.makedirs(model_path_)
        if not os.path.exists(result_path_):
            os.makedirs(result_path_)

        return data_loader_, result_path_, model_path_

    data_loader, result_path, model_path = io_setup()
    loss_tasks = set(loss_tasks)
    loss_tasks.update(auxiliary_loss_tasks)
    settings = replace(settings, loss_tasks=loss_tasks)
    data = data_loader.load(settings.task_data_keys, force_cache_miss=force_cache_miss)
    for index_run in trange(num_runs, desc='Run'):

        output_dir = os.path.join(result_path, 'run_{}'.format(index_run))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_validation_path = os.path.join(output_dir, 'output_validation.npz')
        output_test_path = os.path.join(output_dir, 'output_test.npz')
        if os.path.exists(output_validation_path) and os.path.exists(output_test_path):
            continue

        seed = _seed(settings.seed, index_run, n_gpu)
        data_preparer = DataPreparer(seed, settings.preprocessors, settings.get_split_functions(index_run))
        train_data, validation_data, test_data = make_datasets(data_preparer.prepare(data))

        train(settings, output_validation_path, output_test_path, train_data, validation_data, test_data, n_gpu, device)


def iterate_powerset(items):
    for sub_set in itertools.chain.from_iterable(
            itertools.combinations(items, num) for num in range(1, len(items) + 1)):
        yield sub_set


def named_variations(name):

    erp_tasks = ('epnp', 'pnp', 'elan', 'lan', 'n400', 'p600')
    name = SwitchRemember(name)
    auxiliary_loss_tasks = set()

    if name == 'erp':
        training_variations = list(iterate_powerset(erp_tasks))
        settings = Settings(task_data_keys=(DataKeys.ucl,))
        num_runs = 100
        min_memory = 4 * 1024 ** 3
    elif name == 'erp_joint':
        training_variations = [erp_tasks]
        settings = Settings(task_data_keys=(DataKeys.ucl,))
        num_runs = 100
        min_memory = 4 * 1024 ** 3
    elif name == 'nat_stories':
        training_variations = [('ns_spr',), erp_tasks + ('ns_spr',), erp_tasks]
        settings = Settings(task_data_keys=(DataKeys.natural_stories, DataKeys.ucl))
        num_runs = 100
        min_memory = 4 * 1024 ** 3
    elif name == 'nat_stories_head_loc':
        training_variations = [('ns_spr',), erp_tasks + ('ns_spr',), erp_tasks]
        settings = Settings(
            task_data_keys=(DataKeys.natural_stories, DataKeys.ucl))
        auxiliary_loss_tasks = {'input_head_location'}
        num_runs = 100
        min_memory = 4 * 1024 ** 3
    elif name == 'number_agreement':
        agr = ('colorless', 'linzen_agree')
        training_variations = [agr, erp_tasks + agr, erp_tasks]
        settings = Settings(
            task_data_keys=(DataKeys.colorless_green, DataKeys.linzen_agreement, DataKeys.ucl))
        num_runs = 10
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_fmri':
        training_variations = [('hp_fmri_I',)]
        settings = Settings(
            task_data_keys=(DataKeys.harry_potter,))
        num_runs = 10
        min_memory = 4 * 1024 ** 3
    else:
        raise ValueError('Unknown name: {}. Valid choices are: \n{}'.format(name.var, '\n'.join(name.tests)))

    return training_variations, settings, num_runs, min_memory, auxiliary_loss_tasks


def main():
    parser = argparse.ArgumentParser(
        'Trains a regression by fine-tuning on specified task-variations starting with a BERT pretrained model')

    parser.add_argument('--clean', action='store_true', required=False,
                        help='DANGER: If specified, the current results will'
                             ' be removed and we will start from scratch')

    parser.add_argument('--force_cache_miss', action='store_true', required=False,
                        help='If specified, data will be loaded from raw files and then recached. '
                             'Useful if loading logic has changed')

    parser.add_argument('--log_level', action='store', required=False, default='WARNING',
                        help='Sets the log-level. Defaults to WARNING')

    parser.add_argument('--no_cuda', action='store_true', required=False, help='If specified, model will be run on CPU')

    parser.add_argument(
        '--name', action='store', required=False, default='erp', help='Which set to run')

    args = parser.parse_args()

    logging.getLogger().setLevel(level=args.log_level.upper())

    if args.clean:
        while True:
            answer = input('About to remove results at {}. Is this really what you want to do [y/n]? '.format(
                args.name))
            if answer in {'Y', 'y', 'N', 'n'}:
                if answer == 'N' or answer == 'n':
                    sys.exit(0)
                break

        paths_ = Paths()
        model_path = os.path.join(paths_.model_path, args.name)
        result_path = os.path.join(paths_.result_path, args.name)
        if os.path.exists(model_path):
            rmtree(model_path)
        if os.path.exists(result_path):
            rmtree(result_path)
        sys.exit(0)

    training_variations_, settings_, num_runs_, min_memory_, aux_loss_tasks = named_variations(args.name)
    if args.no_cuda:
        settings_.no_cuda = True
    for training_variation in training_variations_:
        run_variation(args.name, training_variation, settings_, num_runs_, aux_loss_tasks, args.force_cache_miss)


if __name__ == '__main__':
    main()
