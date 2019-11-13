import os
from collections import OrderedDict
import dataclasses
from typing import Mapping, Any, Sequence, Optional
import logging

import numpy as np

"""
Low-level functions for reading raw results from experiment output files
"""


__all__ = ['OutputResult', 'write_predictions', 'read_predictions', 'write_loss_curve', 'read_loss_curve']


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class OutputResult:
    name: str
    critic_type: str
    critic_kwargs: Mapping[str, Any]
    unique_id: int
    data_key: str
    tokens: Sequence[str]
    mask: Sequence[bool]
    prediction: Sequence[float]
    target: Sequence[float]
    sequence_type: str
    word_ids: Optional[Sequence[int]]


def _num_tokens(tokens):
    for idx, token in enumerate(tokens):
        if token == '[PAD]':
            return idx
    return len(tokens)


def write_predictions(output_dir, all_results, data_set, settings):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    """Write final predictions to an output file."""
    logger.info("Writing predictions to: %s" % output_dir)

    for key in all_results:

        output_dict = dict()

        if len(all_results[key]) == 0:
            continue

        critic_settings = settings.get_critic(key, data_set)

        predictions = list()
        targets = list()
        masks = list()
        lengths = list()
        target_lengths = list()
        data_keys = list()
        unique_ids = list()
        tokens = list()
        word_ids = list()

        sequence_type = None
        for detailed_result in all_results[key]:
            if sequence_type is None:
                sequence_type = detailed_result.sequence_type
            else:
                assert(sequence_type == detailed_result.sequence_type)
            current_tokens = data_set.get_tokens(detailed_result.data_set_id, detailed_result.unique_id)
            num_tokens = _num_tokens(current_tokens)
            tokens.extend(current_tokens[:num_tokens])
            unique_ids.append(detailed_result.unique_id)
            data_keys.append(data_set.data_set_key_for_id(detailed_result.data_set_id))
            lengths.append(num_tokens)
            if sequence_type == 'sequence':
                predictions.append(detailed_result.prediction[:num_tokens])
                targets.append(detailed_result.target[:num_tokens])
                if detailed_result.mask is not None:
                    masks.append(detailed_result.mask[:num_tokens])
                else:
                    masks.append(None)
                if detailed_result.word_ids is not None:
                    word_ids.append(detailed_result.word_ids[:num_tokens])
                else:
                    word_ids.append(None)
                target_lengths.append(num_tokens)
            elif sequence_type == 'single':
                predictions.append(np.expand_dims(detailed_result.prediction, 0))
                targets.append(np.expand_dims(detailed_result.target, 0))
                masks.append(np.expand_dims(detailed_result.mask, 0) if detailed_result.mask is not None else None)
                word_ids.append(
                    np.expand_dims(detailed_result.word_ids, 0) if detailed_result.word_ids is not None else None)
            elif sequence_type == 'grouped':
                predictions.append(detailed_result.prediction)
                targets.append(detailed_result.target)
                masks.append(detailed_result.mask)
                word_ids.append(detailed_result.word_ids)
                target_lengths.append(len(detailed_result.target))

        if any(m is None for m in masks) and any(m is not None for m in masks):
            raise ValueError('Unable to write a mixture of None and non-None masks')
        if any(w is None for w in word_ids) and any(w is not None for w in word_ids):
            raise ValueError('Unable to write a mixture of None and non-None word_ids')

        output_dict['predictions'] = np.concatenate(predictions)
        output_dict['target'] = np.concatenate(targets)
        output_dict['masks'] = np.concatenate(masks) if masks[0] is not None else None
        output_dict['lengths'] = np.array(lengths)
        output_dict['target_lengths'] = np.array(target_lengths)
        output_dict['data_keys'] = np.array(data_keys)
        output_dict['unique_ids'] = np.array(unique_ids)
        output_dict['tokens'] = np.array(tokens)
        output_dict['critic'] = critic_settings.critic_type
        output_dict['sequence_type'] = sequence_type
        output_dict['word_ids'] = np.concatenate(word_ids) if word_ids[0] is not None else None
        if critic_settings.critic_kwargs is not None:
            for critic_key in critic_settings.critic_kwargs:
                output_dict['critic_kwarg_{}'.format(critic_key)] = critic_settings.critic_kwargs[critic_key]

        np.savez(os.path.join(output_dir, '{}.npz'.format(key)), **output_dict)

    with open(os.path.join(output_dir, 'keys.txt'), 'wt') as key_file:
        for key in all_results:
            if len(all_results[key]) > 0:
                key_file.write(key)
                key_file.write('\n')


def read_predictions(output_dir, keys=None):

    with open(os.path.join(output_dir, 'keys.txt'), 'rt') as key_file:
        file_keys = [k.strip() for k in key_file.readlines()]
        if keys is None:
            keys = file_keys
        else:
            keys_ = list()
            for key in keys:
                if key not in file_keys:
                    print('Warning: key {} is not available'.format(key))
                else:
                    keys_.append(key)
            keys = keys_

    result = OrderedDict()
    for key in keys:
        with np.load(os.path.join(output_dir, '{}.npz'.format(key)), allow_pickle=True) as npz:

            predictions = npz['predictions']
            target = npz['target']
            masks = npz['masks']
            lengths = npz['lengths']
            target_lengths = npz['target_lengths']
            data_keys = npz['data_keys']
            unique_ids = npz['unique_ids']
            tokens = npz['tokens']
            critic_type = npz['critic'].item()
            sequence_type = npz['sequence_type'].item()
            word_ids = npz['word_ids']

            critic_kwarg_prefix = 'critic_kwarg'
            critic_kwargs = dict()
            for npz_key in npz.keys():
                if npz_key.startswith(critic_kwarg_prefix):
                    critic_kwargs[npz_key[len(critic_kwarg_prefix):]] = npz[npz_key].item()
            if len(critic_kwargs) == 0:
                critic_kwargs = None

            splits = np.cumsum(lengths)[:-1]
            if sequence_type == 'sequence':
                target_splits = splits
            elif sequence_type == 'grouped':
                target_splits = np.cumsum(target_lengths)[:-1]
            else:
                target_splits = None
            if target_splits is not None:
                predictions = np.split(predictions, target_splits)
                target = np.split(target, target_splits)
                if masks is not None:
                    # noinspection PyTypeChecker
                    masks = np.split(masks, target_splits)
                if word_ids is not None:
                    word_ids = np.split(word_ids, target_splits)
            data_keys = [k.item() for k in data_keys]
            unique_ids = [u.item() for u in unique_ids]
            tokens = np.split(tokens, splits)
            tokens = [[t.item() for t in s] for s in tokens]

            results = list()
            for idx in range(len(tokens)):
                results.append(OutputResult(
                    key, critic_type, critic_kwargs,
                    unique_ids[idx], data_keys[idx], tokens[idx], masks[idx], predictions[idx], target[idx],
                    sequence_type, word_ids[idx] if word_ids is not None else None))

            result[key] = results

    return result


def write_loss_curve(output_path, task_results):
    output_dict = dict()
    keys = [k for k in task_results.results]
    output_dict['__keys__'] = keys
    for key in keys:
        output_dict['epochs_{}'.format(key)] = np.array([tr.epoch for tr in task_results.results[key]])
        output_dict['steps_{}'.format(key)] = np.array([tr.step for tr in task_results.results[key]])
        output_dict['values_{}'.format(key)] = np.array([tr.value for tr in task_results.results[key]])
    np.savez(output_path, **output_dict)


def read_loss_curve(output_path):
    npz = np.load(output_path, allow_pickle=True)
    keys = npz['__keys__']
    result = OrderedDict()
    for key in keys:
        result[key] = (npz['epochs_{}'.format(key)], npz['steps_{}'.format(key)], npz['values_{}'.format(key)])
    return result
