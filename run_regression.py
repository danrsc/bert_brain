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
from collections import OrderedDict
import itertools
import logging
import json
import os
import random
from dataclasses import replace, dataclass, asdict as dataclass_as_dict
from typing import Sequence, List
import hashlib
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.optimization import BertAdam

from bert_erp_modeling import BertForTokenRegression, NamedTargetStopWordMSE
from bert_erp_datasets import DataLoader, DataPreparer, make_dataset, max_example_sequence_length, to_device
from bert_erp_settings import Settings
from bert_erp_paths import Paths

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


__all__ = [
    'OutputResult', 'write_predictions', 'task_hash', 'named_variations', 'TaskResult', 'TaskResults', 'evaluate',
    'run_variation', 'SwitchRemember', 'iterate_powerset']


@dataclass
class OutputResult:
    name: str
    tokens: List[str]
    mask: List[int]
    prediction: List[float]
    target: List[float]


def _num_tokens(tokens):
    for idx, token in enumerate(tokens):
        if token == '[PAD]':
            return idx
    return len(tokens)


def write_predictions(output_path, all_results, unique_id_to_tokens):
    """Write final predictions to the json file."""
    logger.info("Writing predictions to: %s" % output_path)

    output_results = list()
    for key in all_results:
        for detailed_result in all_results[key]:
            tokens = unique_id_to_tokens[detailed_result.unique_id]
            num_tokens = _num_tokens(tokens)
            output_results.append(dataclass_as_dict(OutputResult(
                key,
                tokens[:num_tokens],
                [bool(x) for x in detailed_result.mask[:num_tokens]],
                [float(x) for x in detailed_result.prediction[:num_tokens]],
                [float(x) for x in detailed_result.target[:num_tokens]])))
    with open(output_path, "w") as writer:
        writer.write(json.dumps(output_results, indent=4) + "\n")


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan


def task_hash(loss_tasks):
    hash_ = hashlib.sha256()
    for loss_task in sorted(loss_tasks):
        hash_.update(loss_task.encode())
    return hash_.hexdigest()


def _first(mapping):
    for k in mapping:
        return mapping[k]


@dataclass
class TaskResult:
    step: int
    value: float


class TaskResults:

    def __init__(self):
        self.results = OrderedDict()

    def add_result(self, name, step, value):
        if name not in self.results:
            self.results[name] = list()
        self.results[name].append(TaskResult(step, value))


def evaluate(settings, model, loss_handler, device, global_step, eval_results, eval_data, return_detailed=False):

    if settings.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_data_loader = TorchDataLoader(
        eval_data, sampler=eval_sampler, batch_size=settings.predict_batch_size)

    model.eval()
    all_results = OrderedDict()
    logger.info("Start evaluating")

    if settings.show_step_progress:
        batch_iterator = tqdm(eval_data_loader, desc="Evaluating")
    else:
        batch_iterator = eval_data_loader

    total_loss = 0
    total_count = 0
    for input_ids, input_mask, is_stop, is_begin_word_pieces, response_data, unique_ids in batch_iterator:
        # if len(all_results) % 1000 == 0:
        #     logger.info("Processing example: %d" % (len(all_results)))
        input_ids, input_mask, is_stop, is_begin_word_pieces, response_data = to_device(
            device, input_ids, input_mask, is_stop, is_begin_word_pieces, response_data)
        with torch.no_grad():
            predictions = model(input_ids, attention_mask=input_mask)
            loss_result = loss_handler(
                predictions, response_data, is_stop, is_begin_word_pieces,
                return_detailed=return_detailed, unique_ids=unique_ids)
            if return_detailed:
                loss_dict, detailed = loss_result
                for k in detailed:
                    if k not in all_results:
                        all_results[k] = list()
                    all_results[k].extend(detailed[k])
            else:
                loss_dict = loss_result
            loss = None
            for data_key in loss_dict:
                sq_err, valid_count = loss_dict[data_key]
                current = sq_err.detach().cpu().numpy().sum().item() / valid_count
                if data_key in settings.loss_tasks:
                    if loss is None:
                        loss = current
                    else:
                        loss += current
                eval_results.add_result(data_key, global_step, current)
            if loss is not None:
                total_loss += loss
                total_count += len(unique_ids)

    if total_count > 0:
        print('eval', total_loss / total_count)
    else:
        print('eval', np.nan)

    if return_detailed:
        return all_results


def run_variation(set_name, loss_tasks: Sequence[str], settings: Settings, num_runs: int):
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
        data_loader_ = temp_paths.make_data_loader(
            # bert_pre_trained_model_name=settings.bert_model,
            data_key_kwarg_dict={DataLoader.ucl: dict(include_erp=True, include_eye=True, self_paced_inclusion='eye')})
        hash_ = task_hash(loss_tasks)
        model_path_ = os.path.join(temp_paths.model_path, 'bert', set_name, hash_)
        base_path_ = os.path.join(temp_paths.base_path, 'bert', set_name, hash_)

        if not os.path.exists(model_path_):
            os.makedirs(model_path_)
        if not os.path.exists(base_path_):
            os.makedirs(base_path_)

        return data_loader_, base_path_, model_path_

    data_loader, base_path, model_path = io_setup()
    settings = replace(settings, loss_tasks=set(loss_tasks))
    data = data_loader.load(settings.task_data_keys)
    for index_run in trange(num_runs, desc='Run'):
        _run_variation_index(settings, base_path, index_run, data, n_gpu, device)


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


def _run_variation_index(settings: Settings, base_path: str, index_run: int, data, n_gpu, device):

    output_dir = os.path.join(base_path, 'run_{}'.format(index_run))

    output_validation_path = os.path.join(output_dir, 'output_validation.json')
    output_test_path = os.path.join(output_dir, 'output_test.json')
    if os.path.exists(output_validation_path) and os.path.exists(output_test_path):
        return

    seed = _seed(settings.seed, index_run, n_gpu)

    data_preparer = DataPreparer(seed, settings.get_data_preprocessors())
    data = data_preparer.prepare(data)

    max_sequence_length = max_example_sequence_length(data)

    # for now, we are only supporting single dataset here; need to investigate the implications for distributed
    # training etc if we use more than one dataset, but will experiment with one first to see about memory etc.
    assert(len(data) == 1)

    first_data = _first(data)
    train_data, data_key_to_shape, _ = make_dataset(max_sequence_length, first_data.train, first_data.data)
    validation_data, validation_id_to_tokens = None, None
    if first_data.validation is not None and len(first_data.validation) > 0:
        validation_data, _, validation_id_to_tokens = make_dataset(
            max_sequence_length, first_data.validation, first_data.data, include_unique_id=True)

    num_train_steps = int(
        len(first_data.train) /
        settings.train_batch_size /
        settings.gradient_accumulation_steps * settings.num_train_epochs)

    num_predictions = sum(max(1, int(np.prod(data_key_to_shape[k]))) for k in data_key_to_shape)

    # Prepare model
    model = BertForTokenRegression.from_pretrained(settings.bert_model, num_predictions=num_predictions)
    if settings.fp16:
        model.half()
    model.to(device)
    if settings.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[settings.local_rank], output_device=settings.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if settings.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                            for n, param in model.named_parameters()]
    elif settings.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                            for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=settings.learning_rate,
                         warmup=settings.warmup_proportion,
                         t_total=num_train_steps)

    global_step = 0
    train_results = TaskResults()
    validation_results = TaskResults()
    logger.info("***** Running training *****")
    logger.info("  Num orig examples = %d", len(first_data.train))
    logger.info("  Num split examples = %d", len(first_data.train))
    logger.info("  Batch size = %d", settings.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    loss_handler = NamedTargetStopWordMSE(keep_content=True, ordered_shape_dict=data_key_to_shape)

    if settings.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_data_loader = TorchDataLoader(train_data, sampler=train_sampler, batch_size=settings.train_batch_size)

    if settings.show_epoch_progress:
        epoch_range = trange(int(settings.num_train_epochs), desc="Epoch")
    else:
        epoch_range = range(int(settings.num_train_epochs))

    for _ in epoch_range:

        model.train()

        if settings.show_step_progress:
            batch_iterator = tqdm(train_data_loader, desc="Iteration")
        else:
            batch_iterator = train_data_loader

        for step, batch in enumerate(batch_iterator):
            if n_gpu == 1:
                batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
            input_ids, input_mask, is_stop, is_begin_word_pieces, response_data = batch
            predictions = model(input_ids, attention_mask=input_mask)
            loss_dict = loss_handler(predictions, response_data, is_stop, is_begin_word_pieces)
            loss = None
            for data_key in loss_dict:
                sq_err, valid_count = loss_dict[data_key]
                if data_key in settings.loss_tasks:
                    current = sq_err.sum() / valid_count
                    if loss is None:
                        loss = current
                    else:
                        loss += current
                train_results.add_result(
                    data_key,
                    global_step,
                    sq_err.detach().cpu().numpy().sum().item() / valid_count)

            if loss is not None:
                print('train', loss.item() / len(input_ids))
                if n_gpu > 1:  # hmm - not sure how this is supposed to work
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if settings.fp16 and settings.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * settings.loss_scale
                if settings.gradient_accumulation_steps > 1:
                    loss = loss / settings.gradient_accumulation_steps
                loss.backward()
                if (step + 1) % settings.gradient_accumulation_steps == 0:
                    if settings.fp16 or settings.optimize_on_cpu:
                        if settings.fp16 and settings.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                param.grad.data = param.grad.data / settings.loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            settings.loss_scale = settings.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                    model.zero_grad()
                    global_step += 1

        if validation_data is not None:
            evaluate(settings, model, loss_handler, device, global_step, validation_results, validation_data)

    logger.info("***** Running predictions *****")
    logger.info("  Num orig examples = %d", len(first_data.validation))
    logger.info("  Num split examples = %d", len(first_data.validation))
    logger.info("  Batch size = %d", settings.predict_batch_size)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if validation_data is not None:
        all_validation = evaluate(
            settings, model, loss_handler, device, global_step, validation_results, validation_data,
            return_detailed=True)
    else:
        all_validation = []

    test_id_to_tokens = None
    if first_data.test is not None and len(first_data.test) > 0:
        test_data, _, test_id_to_tokens = make_dataset(
            max_sequence_length, first_data.test, first_data.data, include_unique_id=True)
        test_results = TaskResults()
        all_test = evaluate(
            settings, model, loss_handler, device, global_step, test_results, test_data,
            return_detailed=True)
    else:
        all_test = []

    write_predictions(output_validation_path, all_validation, validation_id_to_tokens)
    write_predictions(output_test_path, all_test, test_id_to_tokens)


class SwitchRemember:

    def __init__(self, var):
        self.var = var
        self._tests = set()

    @property
    def tests(self):
        return list(sorted(self._tests))

    def __eq__(self, test):
        self._tests.add(test)
        return self.var == test


def iterate_powerset(items):
    for sub_set in itertools.chain.from_iterable(
            itertools.combinations(items, num) for num in range(1, len(items) + 1)):
        yield sub_set


def named_variations(name):

    erp_tasks = ('epnp', 'pnp', 'elan', 'lan', 'n400', 'p600')
    name = SwitchRemember(name)

    if name == 'erp':
        training_variations = list(iterate_powerset(erp_tasks))
        settings = Settings(task_data_keys=(DataLoader.ucl,))
        num_runs = 100
        min_memory = 4 * 1024 ** 3
    elif name == 'erp_joint':
        training_variations = [erp_tasks]
        settings = Settings(task_data_keys=(DataLoader.ucl,))
        num_runs = 100
        min_memory = 4 * 1024 ** 3
    elif name == 'nat_stories':
        training_variations = [['ns_spr']]
        settings = Settings(task_data_keys=(DataLoader.natural_stories,))
        num_runs = 100
        min_memory = 4 * 1024 ** 3
    else:
        raise ValueError('Unknown name: {}. Valid choices are: \n{}'.format(name.var, '\n'.join(name.tests)))

    return training_variations, settings, num_runs, min_memory


def main():
    parser = argparse.ArgumentParser(
        'Trains a regression by fine-tuning on specified task-variations starting with a BERT pretrained model')

    parser.add_argument('--clean', action='store_true', required=False,
                        help='DANGER: If specified, the current results will'
                             ' be removed and we will start from scratch')

    parser.add_argument(
        '--name', action='store', required=False, default='erp', help='Which set to run')

    args = parser.parse_args()

    if args.clean:
        while True:
            answer = input('About to remove results at {}. Is this really what you want to do [y/n]? '.format(
                args.name))
            if answer in {'Y', 'y', 'N', 'n'}:
                if answer == 'N' or answer == 'n':
                    sys.exit(0)
                break

        paths_ = Paths()
        model_path = os.path.join(paths_.model_path, 'bert', args.name)
        base_path = os.path.join(paths_.base_path, 'bert', args.name)
        if os.path.exists(model_path):
            rmtree(model_path)
        if os.path.exists(base_path):
            rmtree(base_path)
        sys.exit(0)

    training_variations_, settings_, num_runs_, min_memory_ = named_variations(args.name)

    for training_variation in training_variations_:
        run_variation(args.name, training_variation, settings_, num_runs_)


if __name__ == '__main__':
    main()
