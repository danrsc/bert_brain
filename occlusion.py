import os
import itertools
from dataclasses import replace, dataclass
from collections import OrderedDict
from typing import Sequence, Tuple, Mapping

import numpy as np
import torch
from torch.utils.data import SequentialSampler, DataLoader as TorchDataLoader

from bert_erp_datasets import DataPreparer
from bert_erp_settings import Settings
from bert_erp_paths import Paths
from bert_erp_modeling import BertMultiPredictionHead

from train_eval import setup_prediction_heads_and_losses, make_datasets, collate_fn
from run_variations import named_variations, task_hash, set_random_seeds


@dataclass
class OcclusionResult:
    index_run: int
    tokens: Tuple[str, ...]
    index_occluded: int
    loss_dict: Mapping[str, float]
    loss: float


def run_occlusion_for_variation(
            set_name,
            loss_tasks: Sequence[str],
            settings: Settings,
            num_runs: int,
            auxiliary_loss_tasks: Sequence[str]):

    if settings.optimization_settings.local_rank == -1 or settings.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not settings.no_cuda else "cpu")
        n_gpu = 1  # torch.cuda.device_count()
    else:
        device = torch.device("cuda", settings.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if settings.optimization_settings.fp16:
            settings.optimization_settings.fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)

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
    data = corpus_loader.load(settings.corpus_keys)

    tokenizer = corpus_loader.make_bert_tokenizer()
    occlusion_token = '[UNK]'
    occluded_token_id = tokenizer.convert_tokens_to_ids([occlusion_token])[0]

    results = list()

    for index_run in range(num_runs):

        model = BertMultiPredictionHead.load(os.path.join(model_path, 'run_{}'.format(index_run)))
        model.to(device)
        model.eval()

        seed = set_random_seeds(settings.seed, index_run, n_gpu)
        data_preparer = DataPreparer(seed, settings.preprocessors, settings.get_split_functions(index_run))
        _, validation_data, _ = make_datasets(
            data_preparer.prepare(data), data_id_in_batch_keys=settings.data_id_in_batch_keys)

        example_iterator = TorchDataLoader(
            validation_data, sampler=SequentialSampler(validation_data), batch_size=1, collate_fn=collate_fn)

        _, _, _, loss_handlers = setup_prediction_heads_and_losses(settings, validation_data)

        for example in example_iterator:

            for k in example:
                example[k] = example[k].to(device)

            tokens = validation_data.get_tokens(example['data_set_id'].item(), example['unique_id'].item())
            for index_occluded in itertools.chain([-1], range(len(tokens))):

                # shallow copy
                occluded = type(example)((k, example[k]) for k in example)

                if index_occluded >= 0:
                    for k in settings.supplemental_fields:
                        if k in occluded and validation_data.is_sequence(k):
                            occluded[k] = occluded[k].clone()
                            occluded[k][:, index_occluded] = validation_data.fill_value(k)
                    occluded['token_ids'] = occluded['token_ids'].clone()
                    occluded['token_ids'][:, index_occluded] = occluded_token_id
                predictions = model(occluded, validation_data)
                loss_dict = OrderedDict(
                    (h.field,
                     (h.weight,
                      h(occluded, predictions, return_detailed=False, apply_weight=False, as_numpy=True)))
                    for h in loss_handlers)
                loss = None
                for data_key in loss_dict:
                    weight, data_loss = loss_dict[data_key]
                    no_valid_inputs = isinstance(data_loss, str) and data_loss == 'no_valid_inputs'
                    if no_valid_inputs:
                        current = np.nan
                    else:
                        current = data_loss
                    kind = validation_data.response_data_kind(data_key)
                    if (data_key in settings.loss_tasks or kind in settings.loss_tasks) and not no_valid_inputs:
                        if loss is None:
                            loss = weight * current
                        else:
                            loss += weight * current
                results.append(OcclusionResult(index_run, tuple(tokens), index_occluded, loss_dict, loss))

    return results


def run_occlusion(variation_set_name):
    training_variations, settings, num_runs, min_memory, aux_loss_tasks = named_variations(variation_set_name)
    for training_variation in training_variations:
        run_occlusion_for_variation(variation_set_name, training_variation, settings, num_runs, aux_loss_tasks)
