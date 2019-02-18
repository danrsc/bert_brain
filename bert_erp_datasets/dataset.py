import dataclasses
import itertools
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import default_collate

from bert_erp_common import SwitchRemember
from bert_erp_tokenization import RawData, FieldSpec
from .data_preparer import PreparedData


__all__ = ['max_example_sequence_length', 'PreparedDataDataset', 'collate_fn']


def max_example_sequence_length(prepared_or_raw_data):
    result = None
    for k in prepared_or_raw_data:
        if isinstance(prepared_or_raw_data[k], RawData):
            x, y, z = (prepared_or_raw_data[k].input_examples,
                       prepared_or_raw_data[k].validation_input_examples,
                       prepared_or_raw_data[k].test_input_examples)
        elif isinstance(prepared_or_raw_data[k], PreparedData):
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


def collate_fn(batch):
    if isinstance(batch[0], OrderedDict):
        return OrderedDict((k, collate_fn([d[k] for d in batch])) for k in batch[0])
    else:
        return default_collate(batch)


class PreparedDataDataset(torch.utils.data.Dataset):

    @staticmethod
    def _get_examples(which, current):
        which = SwitchRemember(which)
        if which == 'train':
            return current.train
        elif which == 'validation':
            return current.validation
        elif which == 'test':
            return current.test
        raise ValueError(
            'Unknown value for which: {}. Valid choices are: ({})'.format(which.var, ', '.join(which.tests)))

    data_set_id_field = 'data_set_id'

    def __init__(self, max_sequence_length, prepared_data, which='train',
                 token_field='tokens', id_field='unique_id', data_index_field='data_ids'):

        self._num_examples = OrderedDict()
        self._response_data = OrderedDict()
        self._example_tensors = OrderedDict()
        self._data_id_to_tokens = dict()
        self._data_set_id_to_data_set_key = dict()

        field_specs = dict()

        # add a special field to track which data-set
        self._example_tensors[PreparedDataDataset.data_set_id_field] = list()
        field_specs[PreparedDataDataset.data_set_id_field] = FieldSpec(
            fill_value=-1, tensor_dtype=torch.long, is_sequence=False)

        for data_set_id, data_key in enumerate(prepared_data):

            self._data_set_id_to_data_set_key[data_set_id] = data_key

            current = prepared_data[data_key]
            examples = PreparedDataDataset._get_examples(which, current)

            response_data = OrderedDict()

            for key in current.data:
                response_data[key] = list()

            for index_example, f in enumerate(examples):
                if index_example == 0:
                    fields = [field.name for field in dataclasses.fields(f)]
                    for field in fields:

                        if current.field_specs[field].tensor_dtype == str:
                            continue

                        if field not in self._example_tensors:

                            # get the field_spec or create a default
                            if field in field_specs:
                                # could happen if there is a conflict between response_data and example fields
                                raise ValueError('Field name conflict: {}'.format(field))
                            if current.field_specs is not None and field in current.field_specs:
                                field_specs[field] = current.field_specs[field]
                            else:
                                field_specs[field] = FieldSpec()

                            # back-fill this field in case earlier data-sets did not have this feature
                            num_seen = sum(self._num_examples[k] for k in self._num_examples)
                            if field_specs[field].is_sequence:
                                back_fill = _pad(
                                    np.array([field_specs[field].fill_value]), max_sequence_length,
                                    field_specs[field].fill_value)
                            else:
                                back_fill = field_specs[field].fill_value
                            self._example_tensors[field] = [back_fill] * num_seen

                        else:

                            # validate that there is no conflict between the field_spec in the current data-set
                            # and previously seen field_spec
                            if current.field_specs is not None and field in current.field_specs:
                                field_spec = current.field_specs[field]
                            else:
                                field_spec = FieldSpec()
                            if field_spec != field_specs[field]:
                                raise ValueError('FieldSpec conflict on field {}: {}, {}'.format(
                                    field, field_spec, field_specs[field]))

                # add the current example
                example_values = dataclasses.asdict(f)
                for field in self._example_tensors:
                    if field == PreparedDataDataset.data_set_id_field:
                        self._example_tensors[field].append(data_set_id)
                        continue
                    if field in example_values:
                        example_value = example_values[field]
                    else:
                        if field_specs[field].is_sequence:
                            example_value = np.array([field_specs[field].fill_value])
                        else:
                            example_value = field_specs[field].fill_value
                    if field_specs[field].is_sequence:
                        example_value = _pad(example_value, max_sequence_length, field_specs[field].fill_value)
                    self._example_tensors[field].append(example_value)

                # response data is side-information that is indexed into by the example, so it is treated differently
                for key in current.data:
                    if index_example == 0:
                        # make sure that we don't have a naming conflict
                        if key in self._example_tensors:
                            raise ValueError('Field name conflict: {}'.format(key))

                        # get the field_spec or create the default
                        if current.field_specs is not None and key in current.field_specs:
                            field_spec = current.field_specs[key]
                        else:
                            field_spec = FieldSpec()

                        if not field_spec.is_sequence:
                            raise ValueError('response_data fields must be sequences')

                        # make sure it doesn't conflict with a field we have already seen from a different data-set
                        if key in field_specs:
                            # for now, we don't allow different data-sets to use the same field
                            raise ValueError('Field name conflict: {}'.format(key))
                            # if field_spec != field_specs[key]:
                            #     raise ValueError('FieldSpec conflict on field {}: {}, {}'.format(
                            #         key, field_spec, field_specs[key]))
                        else:
                            field_specs[key] = field_spec

                    response_data[key].append(
                        _filled_values(
                            example_values[data_index_field],
                            current.data[key], max_sequence_length, field_specs[key].fill_value))

                # remember the tokens
                self._data_id_to_tokens[(data_set_id, example_values[id_field])] = example_values[token_field]

            if len(examples) > 0:
                for key in response_data:
                    # add a dummy row to use as a no-labels value (when the current example is for a different dataset)
                    response_data[key].append(
                        _filled_values(np.array([-1]), current.data[key], max_sequence_length,
                                       field_specs[key].fill_value))
                    response_data[key] = torch.tensor(response_data[key], dtype=field_specs[key].tensor_dtype)

            self._num_examples[data_key] = len(examples)
            self._response_data[data_key] = response_data

        for key in self._example_tensors:
            self._example_tensors[key] = torch.tensor(
                self._example_tensors[key], dtype=field_specs[key].tensor_dtype)

        self._is_sequence = dict((f, field_specs[f].is_sequence) for f in field_specs)

    def _response_index(self, index):
        cumsum = 0
        for key in self._response_data:
            if index - cumsum < self._num_examples[key]:
                return key, index - cumsum
            cumsum += self._num_examples[key]
        raise IndexError('Index out of bounds: {}'.format(index))

    @property
    def fields(self):
        result = [k for k in self._example_tensors]
        for response_data_key in self._response_data:
            for k in self._response_data[response_data_key]:
                result.append(k)
        return tuple(result)

    def is_response_data(self, field):
        if field not in self._is_sequence:
            raise KeyError('Invalid field: {}'.format(field))
        for response_data_key in self._response_data:
            if field in self._response_data[response_data_key]:
                return True
        return False

    def value_shape(self, field):
        if field not in self._is_sequence:
            raise KeyError('Invalid field: {}'.format(field))
        if field in self._example_tensors:
            size = self._example_tensors[field].size()
            if self._is_sequence[field]:
                return size[2:]
            return size[1:]
        else:
            for response_data_key in self._response_data:
                if field in self._response_data[response_data_key]:
                    size = self._response_data[response_data_key][field].size()
                    if self._is_sequence[field]:
                        return size[2:]
                    return size[1:]

    def __getitem__(self, item):
        result = OrderedDict((k, self._example_tensors[k][item]) for k in self._example_tensors)
        data_key, index_in_response = self._response_index(item)
        for response_data_key in self._response_data:
            i = -1 if data_key != response_data_key else index_in_response
            response_data = self._response_data[response_data_key]
            for k in response_data:
                result[k] = response_data[k][i]
        return result

    def __len__(self):
        for k in self._example_tensors:
            return len(self._example_tensors[k])

    def data_set_key_for_id(self, data_set_id):
        if isinstance(data_set_id, torch.Tensor):
            data_set_id = data_set_id.cpu().item()
        elif isinstance(data_set_id, np.array):
            data_set_id = data_set_id.item()
        return self._data_set_id_to_data_set_key[data_set_id]

    def data_set_key_for_field(self, field):
        for response_data_key in self._response_data:
            if field in self._response_data[response_data_key]:
                return response_data_key
        return None

    def get_tokens(self, data_set_id, item_id):
        if isinstance(data_set_id, torch.Tensor):
            data_set_id = data_set_id.cpu().item()
        elif isinstance(data_set_id, np.array):
            data_set_id = data_set_id.item()
        if isinstance(item_id, torch.Tensor):
            item_id = item_id.cpu().item()
        elif isinstance(item_id, np.array):
            item_id = item_id.item()
        key = (data_set_id, item_id)
        if key not in self._data_id_to_tokens:
            if data_set_id not in self._data_set_id_to_data_set_key:
                raise ValueError('Invalid data_set_id: {}'.format(data_set_id))
            data_set_key = self._data_set_id_to_data_set_key[data_set_id]
            raise KeyError('Item does not exist in dataset: {}, {}'.format(data_set_key, item_id))
        return self._data_id_to_tokens[key]

    def num_examples_for_data_key(self, data_key):
        return self._num_examples[data_key]
