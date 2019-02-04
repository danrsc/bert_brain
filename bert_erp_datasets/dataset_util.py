import itertools
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import TensorDataset

from bert_erp_tokenization import RawData
from .data_preparer import LoadedDataTuple


__all__ = ['max_example_sequence_length', 'make_dataset', 'to_device']


def max_example_sequence_length(prepared_or_raw_data):
    result = None
    for k in prepared_or_raw_data:
        if isinstance(prepared_or_raw_data[k], RawData):
            x, y, z = (prepared_or_raw_data[k].input_examples,
                       prepared_or_raw_data[k].validation_input_examples,
                       prepared_or_raw_data[k].test_input_examples)
        elif isinstance(prepared_or_raw_data[k], LoadedDataTuple):
            x, y, z = prepared_or_raw_data[k].train, prepared_or_raw_data[k].validation, prepared_or_raw_data[k].test
        else:
            raise ValueError('Unexpected type')
        examples = itertools.chain([] if x is None else x, [] if y is None else y, [] if z is None else z)
        current_max = max([len(ex.input_ids) for ex in examples])
        if result is None or current_max > result:
            result = current_max
    return result


def _pad(to_pad, sequence_length, value=0):
    if len(to_pad) < sequence_length:
        return np.pad(to_pad, (0, sequence_length - len(to_pad)), mode='constant', constant_values=value)
    return to_pad


def _pad_tokens(tokens, sequence_length, value='[PAD]'):
    if len(tokens) < sequence_length:
        return list(tokens) + [value] * (sequence_length - len(tokens))
    return list(tokens)


def _filled_values(indices, values, sequence_length, fill_with):
    indices = _pad(indices, sequence_length, value=-1)
    valid_indices = indices[indices >= 0]
    vals = np.full((sequence_length,) + values.shape[1:], fill_with)
    vals[indices >= 0] = values[valid_indices]
    return vals


def make_dataset(max_sequence_length, examples, response_data, include_unique_id=False):
    all_input_ids = torch.tensor([_pad(f.input_ids, max_sequence_length) for f in examples], dtype=torch.long)
    all_input_mask = torch.tensor([_pad(f.input_mask, max_sequence_length) for f in examples], dtype=torch.long)
    all_input_is_stop = torch.tensor(
        [_pad(f.input_is_stop, max_sequence_length, value=1) for f in examples], dtype=torch.uint8)
    all_input_is_begin_word_pieces = torch.tensor(
        [_pad(f.input_is_begin_word_pieces, max_sequence_length) for f in examples], dtype=torch.uint8)

    unique_id_to_tokens = dict()
    for f in examples:
        unique_id_to_tokens[f.unique_id] = _pad_tokens(f.tokens, max_sequence_length)

    data_key_to_shape = OrderedDict()
    response_values = list()
    for key in response_data:
        values = response_data[key]
        data_key_to_shape[key] = values.shape[1:]
        response_values.append(values.reshape(values.shape[0], -1))

    response_values = np.concatenate(response_values, axis=1)

    all_response_values = torch.tensor(
        [_filled_values(f.data_ids, response_values, max_sequence_length, np.nan) for f in examples],
        dtype=torch.float)

    tensors = [all_input_ids, all_input_mask, all_input_is_stop, all_input_is_begin_word_pieces, all_response_values]
    if include_unique_id:
        all_unique_id = torch.tensor([f.unique_id for f in examples], dtype=torch.long)
        tensors.append(all_unique_id)

    return TensorDataset(*tensors), data_key_to_shape, unique_id_to_tokens


def to_device(device, *inputs):
    return tuple(t.to(device) for t in inputs)
