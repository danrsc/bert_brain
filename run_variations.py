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
from bert_erp_datasets import CorpusKeys, DataPreparer, ResponseKind
from bert_erp_settings import Settings, OptimizationSettings, PredictionHeadSettings
from bert_erp_paths import Paths
from bert_erp_modeling import KeyedLinear
from train_eval import train, make_datasets

__all__ = ['task_hash', 'named_variations', 'run_variation', 'iterate_powerset', 'set_random_seeds']


replace_root_logger_handler()
logger = logging.getLogger(__name__)


def task_hash(loss_tasks):
    hash_ = hashlib.sha256()
    for loss_task in sorted(loss_tasks):
        hash_.update(loss_task.encode())
    return hash_.hexdigest()


def set_random_seeds(seed, index_run, n_gpu):
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

    if settings.optimization_settings.local_rank == -1 or settings.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not settings.no_cuda else "cpu")
        n_gpu = 1  # torch.cuda.device_count()
    else:
        device = torch.device("cuda", settings.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if settings.optimization_settings.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            settings.optimization_settings.fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits trainiing: {}".format(
        device, n_gpu, bool(settings.optimization_settings.local_rank != -1), settings.optimization_settings.fp16))

    if settings.optimization_settings.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            settings.optimization_settings.gradient_accumulation_steps))

    # TODO: seems like this is taken care of below?
    # settings = replace(
    #   settings, train_batch_size=int(settings.train_batch_size / settings.gradient_accumulation_steps))

    def io_setup():
        temp_paths = Paths()
        corpus_loader_ = temp_paths.make_corpus_loader(corpus_key_kwarg_dict=settings.corpus_key_kwargs)
        hash_ = task_hash(loss_tasks)
        model_path_ = os.path.join(temp_paths.model_path, set_name, hash_)
        result_path_ = os.path.join(temp_paths.result_path, set_name, hash_)

        if not os.path.exists(model_path_):
            os.makedirs(model_path_)
        if not os.path.exists(result_path_):
            os.makedirs(result_path_)

        return corpus_loader_, result_path_, model_path_

    corpus_loader, result_path, model_path = io_setup()
    loss_tasks = set(loss_tasks)
    loss_tasks.update(auxiliary_loss_tasks)
    settings = replace(settings, loss_tasks=loss_tasks)
    data = corpus_loader.load(settings.corpus_keys, force_cache_miss=force_cache_miss)
    for index_run in trange(num_runs, desc='Run'):

        output_dir = os.path.join(result_path, 'run_{}'.format(index_run))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_validation_path = os.path.join(output_dir, 'output_validation.npz')
        output_test_path = os.path.join(output_dir, 'output_test.npz')
        if os.path.exists(output_validation_path) and os.path.exists(output_test_path):
            continue

        output_model_path = os.path.join(model_path, 'run_{}'.format(index_run))

        seed = set_random_seeds(settings.seed, index_run, n_gpu)
        data_preparer = DataPreparer(seed, settings.preprocessors, settings.get_split_functions(index_run))
        train_data, validation_data, test_data = make_datasets(
            data_preparer.prepare(data),
            data_id_in_batch_keys=settings.data_id_in_batch_keys)

        train(settings, output_validation_path, output_test_path, output_model_path,
              train_data, validation_data, test_data, n_gpu, device)


def iterate_powerset(items):
    for sub_set in itertools.chain.from_iterable(
            itertools.combinations(items, num) for num in range(1, len(items) + 1)):
        yield sub_set


def named_variations(name):

    erp_tasks = ('epnp', 'pnp', 'elan', 'lan', 'n400', 'p600')
    ns_froi_tasks = ('ns_lh_pt', 'ns_lh_at', 'ns_lh_ifg', 'ns_lh_ifgpo', 'ns_lh_mfg', 'ns_lh_ag',
                     'ns_rh_pt', 'ns_rh_at', 'ns_rh_ifg', 'ns_rh_ifgpo', 'ns_rh_mfg', 'ns_rh_ag')

    name = SwitchRemember(name)
    auxiliary_loss_tasks = set()

    if name == 'erp':
        training_variations = list(iterate_powerset(erp_tasks))
        settings = Settings(corpus_keys=(CorpusKeys.ucl,))
        num_runs = 100
        min_memory = 4 * 1024 ** 3
    elif name == 'erp_joint':
        training_variations = [erp_tasks]
        settings = Settings(corpus_keys=(CorpusKeys.ucl,))
        num_runs = 100
        min_memory = 4 * 1024 ** 3
    elif name == 'nat_stories':
        training_variations = [('ns_spr',),
                               erp_tasks + ('ns_spr',),
                               ns_froi_tasks + ('ns_spr',),
                               erp_tasks + ns_froi_tasks,
                               erp_tasks + ns_froi_tasks + ('ns_spr',)]
        settings = Settings(
            corpus_keys=(CorpusKeys.natural_stories, CorpusKeys.ucl),
            optimization_settings=OptimizationSettings(num_train_epochs=50))
        settings.prediction_heads[ResponseKind.ns_froi] = PredictionHeadSettings(
            ResponseKind.ns_froi, KeyedLinear, dict(is_sequence=False))
        settings.corpus_key_kwargs[CorpusKeys.natural_stories] = dict(
            froi_window_duration=10.,
            froi_minimum_duration_required=9.5,
            froi_use_word_unit_durations=False,
            froi_sentence_mode='ignore')
        num_runs = 10
        min_memory = 4 * 1024 ** 3
    elif name == 'ns_froi':
        training_variations = [ns_froi_tasks]
        settings = Settings(
            corpus_keys=(CorpusKeys.natural_stories,),
            optimization_settings=OptimizationSettings(num_train_epochs=3))
        settings.prediction_heads[ResponseKind.ns_froi] = PredictionHeadSettings(
            ResponseKind.ns_froi, KeyedLinear, dict(is_sequence=False))
        settings.corpus_key_kwargs[CorpusKeys.natural_stories] = dict(
            froi_sentence_mode='ignore',
            froi_window_duration=10.,
            froi_minimum_duration_required=9.5,
            froi_use_word_unit_durations=False,
            froi_minimum_story_count=2,
            include_reaction_times=False)
        num_runs = 7  # there are 7 stories that have been recorded in froi
        min_memory = 4 * 1024 ** 3
    elif name == 'ns_hp':
        training_variations = [('hp_fmri_I',),
                               ns_froi_tasks,
                               ns_froi_tasks + ('hp_fmri_I',)]
        settings = Settings(
            corpus_keys=(CorpusKeys.natural_stories, CorpusKeys.harry_potter),
            optimization_settings=OptimizationSettings(num_train_epochs=50))
        settings.prediction_heads[ResponseKind.hp_fmri] = PredictionHeadSettings(
            ResponseKind.hp_fmri, KeyedLinear, dict(is_sequence=False))
        settings.prediction_heads[ResponseKind.ns_froi] = PredictionHeadSettings(
            ResponseKind.ns_froi, KeyedLinear, dict(is_sequence=False))
        settings.corpus_key_kwargs[CorpusKeys.natural_stories] = dict(
            froi_sentence_mode='ignore',
            froi_window_duration=10.,
            froi_minimum_duration_required=9.5,
            froi_use_word_unit_durations=False,
            froi_minimum_story_count=2,
            include_reaction_times=False)
        settings.corpus_key_kwargs[CorpusKeys.harry_potter] = dict(
            fmri_subjects='I',
            fmri_sentence_mode='ignore',
            fmri_window_duration=10.,
            fmri_minimum_duration_required=9.5)
        num_runs = 10
        min_memory = 4 * 1024 ** 3
    elif name == 'nat_stories_head_loc':
        training_variations = [('ns_spr',), erp_tasks + ('ns_spr',), erp_tasks]
        settings = Settings(corpus_keys=(CorpusKeys.natural_stories, CorpusKeys.ucl))
        auxiliary_loss_tasks = {'input_head_location'}
        num_runs = 100
        min_memory = 4 * 1024 ** 3
    elif name == 'number_agreement':
        agr = ('colorless', 'linzen_agree')
        training_variations = [agr, erp_tasks + agr, erp_tasks]
        settings = Settings(
            corpus_keys=(CorpusKeys.colorless_green, CorpusKeys.linzen_agreement, CorpusKeys.ucl),
            optimization_settings=OptimizationSettings(num_train_epochs=50))
        num_runs = 10
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_fmri':
        training_variations = [('hp_fmri_I',)]
        settings = Settings(
            corpus_keys=(CorpusKeys.harry_potter,),
            optimization_settings=OptimizationSettings(num_train_epochs=100))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_fmri_20':
        training_variations = [('hp_fmri_I',)]
        settings = Settings(
            corpus_keys=(CorpusKeys.harry_potter,),
            optimization_settings=OptimizationSettings(num_train_epochs=50))
        settings.prediction_heads[ResponseKind.hp_fmri] = PredictionHeadSettings(
            ResponseKind.hp_fmri, KeyedLinear, dict(is_sequence=False))
        settings.corpus_key_kwargs[CorpusKeys.harry_potter] = dict(
            fmri_subjects='I',
            fmri_sentence_mode='ignore', fmri_window_duration=10., fmri_minimum_duration_required=9.5)
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_fmri_meg':
        training_variations = [('hp_fmri_I', 'hp_meg'), ('hp_meg',), ('hp_fmri_I',)]
        settings = Settings(
            corpus_keys=(CorpusKeys.harry_potter,),
            optimization_settings=OptimizationSettings(num_train_epochs=50))
        settings.prediction_heads[ResponseKind.hp_fmri] = PredictionHeadSettings(
            ResponseKind.hp_fmri, KeyedLinear, dict(is_sequence=False))
        settings.corpus_key_kwargs[CorpusKeys.harry_potter] = dict(
            fmri_subjects='I',
            fmri_sentence_mode='ignore',
            fmri_window_duration=10.,
            fmri_minimum_duration_required=9.5,
            group_meg_sentences_like_fmri=True,
            use_pca_meg=False)
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_fmri_20_linear':
        training_variations = [('hp_fmri_I',)]
        settings = Settings(
            corpus_keys=(CorpusKeys.harry_potter,),
            optimization_settings=OptimizationSettings(num_train_epochs=50, is_train_prediction_heads_only=True))
        settings.prediction_heads[ResponseKind.hp_fmri] = PredictionHeadSettings(
            ResponseKind.hp_fmri, KeyedLinear, dict(is_sequence=False))
        settings.corpus_key_kwargs[CorpusKeys.harry_potter] = dict(
            fmri_subjects='I',
            fmri_sentence_mode='ignore', fmri_window_duration=10., fmri_minimum_duration_required=9.5)
        num_runs = 4
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
                    print('No action taken')
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
