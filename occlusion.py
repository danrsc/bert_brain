import logging
import os
import itertools
from dataclasses import replace, dataclass
from collections import OrderedDict
from typing import Sequence, Any, Mapping, Union
from tqdm import trange
from tqdm_logging import replace_root_logger_handler

import numpy as np
import torch
from torch.utils.data import SequentialSampler, DataLoader as TorchDataLoader

from bert_erp_common import cuda_most_free_device
from bert_erp_datasets import DataPreparer
from bert_erp_settings import Settings
from bert_erp_paths import Paths
from bert_erp_modeling import BertMultiPredictionHead

from train_eval import setup_prediction_heads_and_losses, make_datasets, collate_fn
from run_variations import named_variations, task_hash, set_random_seeds


replace_root_logger_handler()
logger = logging.getLogger(__name__)
logger.setLevel('INFO')


@dataclass
class OcclusionResult:
    name: str
    critic_type: str
    critic_kwargs: Mapping[str, Any]
    unique_id: int
    data_key: str
    tokens: Sequence[str]
    mask: np.ndarray
    prediction: Sequence[Union[np.ndarray, Sequence[np.ndarray]]]
    target: Union[np.ndarray, Sequence[np.ndarray]]
    sequence_type: str


def _num_tokens(tokens):
    for idx, token in enumerate(tokens):
        if token == '[PAD]':
            return idx
    return len(tokens)


def _run_occlusion_for_variation(model_path, data, tokenizer, settings: Settings, index_run: int, device, n_gpu):

    occlusion_token = '[UNK]'
    occluded_token_id = tokenizer.convert_tokens_to_ids([occlusion_token])[0]

    all_results = OrderedDict()

    model = BertMultiPredictionHead.load(
        os.path.join(model_path, 'run_{}'.format(index_run)),
        map_location=lambda storage, loc: None if loc == 'cpu' else storage.cuda(device.index))
    model.to(device)
    model.eval()

    seed = set_random_seeds(settings.seed, index_run, n_gpu)
    data_preparer = DataPreparer(seed, settings.preprocessors, settings.get_split_functions(index_run))
    _, validation_data, _ = make_datasets(
        data_preparer.prepare(data), data_id_in_batch_keys=settings.data_id_in_batch_keys)

    batch_iterator = TorchDataLoader(
        validation_data,
        sampler=SequentialSampler(validation_data),
        batch_size=settings.optimization_settings.predict_batch_size,
        collate_fn=collate_fn)

    _, _, _, loss_handlers = setup_prediction_heads_and_losses(settings, validation_data)

    for batch in batch_iterator:

        # first determine the tokens
        data_set_ids = batch['data_set_id'].cpu().numpy()
        unique_ids = batch['unique_id'].cpu().numpy()

        max_sequence_in_batch = max(
            _num_tokens(validation_data.get_tokens(d, u)) for d, u in zip(data_set_ids, unique_ids))

        for k in batch:
            batch[k] = batch[k].to(device)

        for index_occluded in itertools.chain([-1], range(max_sequence_in_batch)):

            with torch.no_grad():

                # shallow copy
                occluded = type(batch)((k, batch[k]) for k in batch)

                if index_occluded >= 0:
                    for k in settings.supplemental_fields:
                        if k in occluded and validation_data.is_sequence(k):
                            occluded[k] = occluded[k].clone()
                            occluded[k][:, index_occluded] = validation_data.fill_value(k)
                    occluded['token_ids'] = occluded['token_ids'].clone()
                    occluded['token_ids'][:, index_occluded] = occluded_token_id

                predictions = model(occluded, validation_data)
                loss_result = OrderedDict(
                    (h.field,
                     (h.weight,
                      h(occluded, predictions, return_detailed=True, apply_weight=False, as_numpy=True)))
                    for h in loss_handlers)
                for k in loss_result:
                    weight, (summary, detailed) = loss_result[k]
                    if k not in all_results:
                        all_results[k] = OrderedDict()
                    for detailed_result in detailed:

                        current_tokens = validation_data.get_tokens(
                            detailed_result.data_set_id, detailed_result.unique_id)
                        current_tokens = current_tokens[:_num_tokens(current_tokens)]

                        if index_occluded >= len(current_tokens):
                            continue

                        result_key = (detailed_result.data_set_id.cpu().item(), detailed_result.unique_id.cpu().item())
                        if result_key not in all_results[k]:
                            critic_settings = settings.get_critic(k, validation_data)
                            all_results[k][result_key] = OcclusionResult(
                                k,
                                critic_settings.critic_type,
                                critic_settings.critic_kwargs,
                                detailed_result.unique_id,
                                validation_data.data_set_key_for_id(detailed_result.data_set_id),
                                current_tokens,
                                detailed_result.mask,
                                list(),
                                detailed_result.target,
                                detailed_result.sequence_type)
                        all_results[k][result_key].prediction.append(detailed_result.prediction)

    return all_results


def run_occlusion(variation_set_name):

    def io_setup():
        temp_paths = Paths()
        corpus_loader_ = temp_paths.make_corpus_loader(corpus_key_kwarg_dict=settings.corpus_key_kwargs)
        hash_ = task_hash(training_variation)
        model_path_ = os.path.join(temp_paths.model_path, variation_set_name, hash_)
        result_path_ = os.path.join(temp_paths.result_path, variation_set_name, hash_)

        if not os.path.exists(model_path_):
            os.makedirs(model_path_)
        if not os.path.exists(result_path_):
            os.makedirs(result_path_)

        return corpus_loader_, result_path_, model_path_

    training_variations, settings, num_runs, min_memory, aux_loss_tasks = named_variations(variation_set_name)

    if settings.optimization_settings.local_rank == -1 or settings.no_cuda:
        if not torch.cuda.is_available or settings.no_cuda:
            device = torch.device('cpu')
        else:
            device_id, free = cuda_most_free_device()
            torch.cuda.set_device(device_id)
            logger.info('binding to device {} with {} memory free'.format(device_id, free))
            device = torch.device('cuda:{}'.format(device_id))
        n_gpu = 1  # torch.cuda.device_count()
    else:
        device = torch.device('cuda', settings.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if settings.optimization_settings.fp16:
            settings.optimization_settings.fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)

    for training_variation in training_variations:

        print('Running on variation: {}'.format(training_variation))

        corpus_loader, result_path, model_path = io_setup()
        loss_tasks = set(training_variation)
        loss_tasks.update(aux_loss_tasks)
        settings = replace(settings, loss_tasks=loss_tasks)
        data = corpus_loader.load(settings.corpus_keys)
        tokenizer = corpus_loader.make_bert_tokenizer()

        for index_run in trange(num_runs, desc='Runs'):
            run_results = _run_occlusion_for_variation(model_path, data, tokenizer, settings, index_run, device, n_gpu)
            write_occlusion_predictions(
                os.path.join(result_path, 'run_{}'.format(index_run), 'output_validation_occlusion.npz'),
                run_results)

    print('Done')


def write_occlusion_predictions(output_path, all_results):

    """Write final predictions to an output file."""
    logger.info("Writing predictions to: %s" % output_path)

    output_dict = dict()
    for key in all_results:

        predictions = list()
        targets = list()
        masks = list()
        lengths = list()
        num_occlusions = list()
        target_lengths = list()
        data_keys = list()
        unique_ids = list()
        tokens = list()

        sequence_type = None
        critic_type = None
        critic_kwargs = None
        for detailed_result_key in all_results[key]:
            detailed_result = all_results[key][detailed_result_key]
            if sequence_type is None:
                sequence_type = detailed_result.sequence_type
                critic_type = detailed_result.critic_type
                critic_kwargs = detailed_result.critic_kwargs
            else:
                assert(sequence_type == detailed_result.sequence_type)
                assert((critic_kwargs is None) == (detailed_result.critic_kwargs is None))
                if critic_kwargs is not None:
                    assert(len(critic_kwargs) == len(detailed_result.critic_kwargs))
                    assert(all(k in detailed_result.critic_kwargs for k in critic_kwargs))
                    assert(all(critic_kwargs[k] == detailed_result.critic_kwargs[k] for k in critic_kwargs))
                assert(critic_type == detailed_result.critic_type)

            num_tokens = len(detailed_result.tokens)
            tokens.extend(detailed_result.tokens)
            unique_ids.append(detailed_result.unique_id)
            data_keys.append(detailed_result.data_key)
            lengths.append(len(detailed_result.tokens))
            num_occlusions.append(len(detailed_result.prediction))
            if sequence_type == 'sequence':
                for p in detailed_result.prediction:
                    predictions.append(p[:num_tokens])
                targets.append(detailed_result.target[:num_tokens])
                if detailed_result.mask is not None:
                    masks.append(detailed_result.mask[:num_tokens])
                else:
                    masks.append(None)
                target_lengths.append(num_tokens)
            elif sequence_type == 'single':
                for p in detailed_result.prediction:
                    predictions.append(np.expand_dims(p, 0))
                targets.append(np.expand_dims(detailed_result.target, 0))
                masks.append(np.expand_dims(detailed_result.mask, 0) if detailed_result.mask is not None else None)
            elif sequence_type == 'grouped':
                for p in detailed_result.prediction:
                    predictions.append(p)
                targets.append(detailed_result.target)
                masks.append(detailed_result.mask)
                target_lengths.append(len(detailed_result.target))

        if any(m is None for m in masks) and any(m is not None for m in masks):
            raise ValueError('Unable to write a mixture of None and non-None masks')

        output_dict['predictions_{}'.format(key)] = np.concatenate(predictions)
        output_dict['target_{}'.format(key)] = np.concatenate(targets)
        output_dict['masks_{}'.format(key)] = np.concatenate(masks) if masks[0] is not None else None
        output_dict['lengths_{}'.format(key)] = np.array(lengths)
        output_dict['num_occlusions_{}'.format(key)] = np.array(num_occlusions)
        output_dict['target_lengths_{}'.format(key)] = np.array(target_lengths)
        output_dict['data_keys_{}'.format(key)] = np.array(data_keys)
        output_dict['unique_ids_{}'.format(key)] = np.array(unique_ids)
        output_dict['tokens_{}'.format(key)] = np.array(tokens)
        output_dict['critic_{}'.format(key)] = critic_type
        output_dict['sequence_type_{}'.format(key)] = sequence_type
        if critic_kwargs is not None:
            for critic_key in critic_kwargs:
                output_dict['critic_kwarg_{}_{}'.format(key, critic_key)] = critic_kwargs[critic_key]

    np.savez(output_path, keys=np.array([k for k in all_results]), **output_dict)


def read_occlusion_predictions(output_path):
    npz = np.load(output_path)

    keys = [k.item() for k in npz['keys']]

    result = OrderedDict()
    for key in keys:
        predictions = npz['predictions_{}'.format(key)]
        target = npz['target_{}'.format(key)]
        masks = npz['masks_{}'.format(key)]
        lengths = npz['lengths_{}'.format(key)]
        num_occlusions = npz['num_occlusions_{}'.format(key)]
        assert(len(lengths) == len(num_occlusions))
        target_lengths = npz['target_lengths_{}'.format(key)]
        data_keys = npz['data_keys_{}'.format(key)]
        unique_ids = npz['unique_ids_{}'.format(key)]
        tokens = npz['tokens_{}'.format(key)]
        critic_type = npz['critic_{}'.format(key)].item()
        sequence_type = npz['sequence_type_{}'.format(key)].item()
        critic_kwarg_prefix = 'critic_kwarg_{}'.format(key)
        critic_kwargs = dict()
        for npz_key in npz.keys():
            if npz_key.startswith(critic_kwarg_prefix):
                critic_kwargs[npz_key[len(critic_kwarg_prefix):]] = npz[npz_key].item()
        if len(critic_kwargs) == 0:
            critic_kwargs = None

        splits = np.cumsum(lengths)[:-1]
        if sequence_type == 'sequence':
            target_splits = splits
            target_length_info = lengths
            prediction_splits = np.cumsum([length * occ for length, occ in zip(lengths, num_occlusions)])[:-1]
        elif sequence_type == 'grouped':
            target_splits = np.cumsum(target_lengths)[:-1]
            target_length_info = target_lengths
            prediction_splits = np.cumsum([length * occ for length, occ in zip(target_lengths, num_occlusions)])[:-1]
        else:
            target_splits = None
            target_length_info = None
            prediction_splits = np.cumsum(num_occlusions)[:-1]
        if target_splits is not None:
            target = np.split(target, target_splits)
            if masks is not None:
                masks = np.split(masks, target_splits)
        predictions = np.split(predictions, prediction_splits)
        for index_prediction in range(len(predictions)):

            reshape_shape = (num_occlusions[index_prediction],)
            if target_length_info is not None:
                reshape_shape = reshape_shape + (target_length_info[index_prediction],)
            reshape_shape = reshape_shape + predictions[index_prediction].shape[1:]
            predictions[index_prediction] = np.reshape(predictions[index_prediction], reshape_shape)

        data_keys = [k.item() for k in data_keys]
        unique_ids = [u.item() for u in unique_ids]
        tokens = np.split(tokens, splits)
        tokens = [[t.item() for t in s] for s in tokens]

        results = list()
        for idx in range(len(tokens)):
            results.append(OcclusionResult(
                key, critic_type, critic_kwargs,
                unique_ids[idx], data_keys[idx], tokens[idx], masks[idx], predictions[idx], target[idx], sequence_type))

        result[key] = results

    return result


@dataclass
class OcclusionSensitivity:
    name: str
    unique_id: int
    data_key: str
    tokens: Sequence[str]
    sensitivity: Sequence[float]


def sensitivity_delta_mse(prediction, target):
    sq_err = np.square(target - prediction)
    # the 0th item is non-occluded;
    # take the squared diff between it (using slice to keep the 1st axis) and each other item
    return np.nanmean(sq_err[:1] - sq_err[1:], axis=1)


def occlusion_sensitivity(occlusion_results, mask=None, sensitivity_fn=None):
    indices = np.where(np.reshape(mask, -1))[0]
    sensitivities = list()
    min_sensitivity = None
    max_sensitivity = None
    for occlusion_result in occlusion_results:
        target = np.reshape(occlusion_result.target, (1, -1))[:, indices]
        prediction = np.reshape(occlusion_result.prediction, (occlusion_result.prediction.shape[0], -1))[:, indices]
        if sensitivity_fn is None:
            sq_err = np.square(target - prediction)
            # the 0th item is non-occluded;
            # take the squared diff between it (using slice to keep the 1st axis) and each other item
            sensitivity = np.nanmean(np.square(sq_err[:1] - sq_err[1:]), axis=1)
            sensitivity = sensitivity / np.sum(sensitivity, keepdims=True)
        else:
            sensitivity = sensitivity_fn(prediction, target)
        current_min = np.nanmin(sensitivity)
        current_max = np.nanmax(sensitivity)
        if min_sensitivity is None:
            min_sensitivity, max_sensitivity = current_min, current_max
        else:
            min_sensitivity = min(min_sensitivity, current_min)
            max_sensitivity = max(max_sensitivity, current_max)
        sensitivities.append(
            OcclusionSensitivity(
                occlusion_result.name,
                occlusion_result.unique_id,
                occlusion_result.data_key,
                occlusion_result.tokens,
                sensitivity))
    return sensitivities, min_sensitivity, max_sensitivity
