from collections import OrderedDict
import dataclasses
from typing import Mapping, Any, Sequence
import logging

import numpy as np


__all__ = ['OutputResult', 'write_predictions', 'read_predictions']


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


def _num_tokens(tokens):
    for idx, token in enumerate(tokens):
        if token == '[PAD]':
            return idx
    return len(tokens)


def write_predictions(output_path, all_results, data_set, settings):

    """Write final predictions to an output file."""
    logger.info("Writing predictions to: %s" % output_path)

    output_dict = dict()
    for key in all_results:

        critic_type = 'mse'
        critic_kwargs = None
        if key in settings.task_settings:
            critic_type = settings.task_settings[key].critic_type
            critic_kwargs = settings.task_settings[key].critic_kwargs
        else:
            task_owner_data_key = data_set.data_set_key_for_field(key)
            if task_owner_data_key is not None and task_owner_data_key in settings.task_settings:
                critic_type = settings.task_settings[task_owner_data_key].critic_type
                critic_kwargs = settings.task_settings[task_owner_data_key].critic_kwargs

        is_sequence = data_set.is_sequence(key)

        predictions = list()
        targets = list()
        masks = list()
        lengths = list()
        data_keys = list()
        unique_ids = list()
        tokens = list()

        for detailed_result in all_results[key]:
            current_tokens = data_set.get_tokens(detailed_result.data_set_id, detailed_result.unique_id)
            num_tokens = _num_tokens(current_tokens)
            tokens.extend(current_tokens[:num_tokens])
            unique_ids.append(detailed_result.unique_id)
            data_keys.append(data_set.data_set_key_for_id(detailed_result.data_set_id))
            lengths.append(num_tokens)
            if is_sequence:
                predictions.append(detailed_result.prediction[:num_tokens])
                targets.append(detailed_result.target[:num_tokens])
                if detailed_result.mask is not None:
                    masks.append(detailed_result.mask[:num_tokens])
                else:
                    masks.append(None)
            else:
                predictions.append(np.expand_dims(detailed_result.prediction, 0))
                targets.append(np.expand_dims(detailed_result.target, 0))
                masks.append(np.expand_dims(detailed_result.mask, 0) if detailed_result.mask is not None else None)

        if any(m is None for m in masks) and any(m is not None for m in masks):
            raise ValueError('Unable to write a mixture of None and non-None masks')

        output_dict['predictions_{}'.format(key)] = np.concatenate(predictions)
        output_dict['target_{}'.format(key)] = np.concatenate(targets)
        output_dict['masks_{}'.format(key)] = np.concatenate(masks) if masks[0] is not None else None
        output_dict['lengths_{}'.format(key)] = np.array(lengths)
        output_dict['data_keys_{}'.format(key)] = np.array(data_keys)
        output_dict['unique_ids_{}'.format(key)] = np.array(unique_ids)
        output_dict['tokens_{}'.format(key)] = np.array(tokens)
        output_dict['critic_{}'.format(key)] = critic_type
        output_dict['is_sequence_{}'.format(key)] = is_sequence
        if critic_kwargs is not None:
            for critic_key in critic_kwargs:
                output_dict['critic_kwarg_{}_{}'.format(key, critic_key)] = critic_kwargs[critic_key]

    np.savez(output_path, keys=np.array([k for k in all_results]), **output_dict)


def read_predictions(output_path):
    npz = np.load(output_path)

    keys = [k.item() for k in npz['keys']]

    result = OrderedDict()
    for key in keys:
        predictions = npz['predictions_{}'.format(key)]
        target = npz['target_{}'.format(key)]
        masks = npz['masks_{}'.format(key)]
        lengths = npz['lengths_{}'.format(key)]
        data_keys = npz['data_keys_{}'.format(key)]
        unique_ids = npz['unique_ids_{}'.format(key)]
        tokens = npz['tokens_{}'.format(key)]
        critic_type = npz['critic_{}'.format(key)].item()
        is_sequence = npz['is_sequence_{}'.format(key)].item()
        critic_kwarg_prefix = 'critic_kwarg_{}'.format(key)
        critic_kwargs = dict()
        for npz_key in npz.keys():
            if npz_key.startswith(critic_kwarg_prefix):
                critic_kwargs[npz_key[len(critic_kwarg_prefix):]] = npz[npz_key].item()
        if len(critic_kwargs) == 0:
            critic_kwargs = None

        splits = np.cumsum(lengths)[:-1]
        if is_sequence:
            predictions = np.split(predictions, splits)
            target = np.split(target, splits)
            if masks is not None:
                masks = np.split(masks, splits)
        data_keys = [k.item() for k in data_keys]
        unique_ids = [u.item() for u in unique_ids]
        tokens = np.split(tokens, splits)
        tokens = [[t.item() for t in s] for s in tokens]

        results = list()
        for idx in range(len(tokens)):
            results.append(OutputResult(
                key, critic_type, critic_kwargs,
                unique_ids[idx], data_keys[idx], tokens[idx], masks[idx], predictions[idx], target[idx]))

        result[key] = results

    return result
