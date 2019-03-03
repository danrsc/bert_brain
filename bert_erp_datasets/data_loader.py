import os
from collections import OrderedDict
import itertools
import dataclasses

import numpy as np
import torch

from pytorch_pretrained_bert import BertTokenizer

from bert_erp_tokenization import InputFeatures, RawData, make_tokenizer_model, FieldSpec
from .university_college_london_corpus import ucl_data
from .natural_stories import natural_stories_data
from .harry_potter import harry_potter_data
from .number_dataset import number_agreement_data


__all__ = ['DataLoader']


def _save_to_cache(cache_path, data, kwargs):

    cache_dir, _ = os.path.split(cache_path)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    num_input_examples = 0 if data.input_examples is None else len(data.input_examples)
    has_input_examples = data.input_examples is not None
    num_validation_input_examples = 0 if data.validation_input_examples is None else len(data.validation_input_examples)
    has_validation_input_examples = data.validation_input_examples is not None
    num_test_input_examples = 0 if data.test_input_examples is None else len(data.test_input_examples)
    has_test_input_examples = data.test_input_examples is not None

    all_examples = list()
    if data.input_examples is not None:
        all_examples.extend(data.input_examples)
    if data.validation_input_examples is not None:
        all_examples.extend(data.validation_input_examples)
    if data.test_input_examples is not None:
        all_examples.extend(data.test_input_examples)

    result = dict((k, list()) for k in dataclasses.asdict(all_examples[0]))
    result['__lengths__'] = list()
    for example in all_examples:
        ex = dataclasses.asdict(example)
        for k in ex:
            if data.field_specs is not None and k in data.field_specs and not data.field_specs[k].is_sequence:
                result[k].append(ex[k])
            elif k == 'tokens':
                result['__lengths__'].append(len(ex[k]))
                result[k].extend(ex[k])
            else:
                result[k].extend(ex[k])

    for k in result:
        result[k] = np.array(result[k])

    for k in kwargs:
        result['__kwarg__{}'.format(k)] = kwargs[k]

    for k in data.response_data:
        result['__response_data__{}'.format(k)] = data.response_data[k]

    if data.field_specs is not None:
        for k in data.field_specs:
            tensor_dtype = 'str' if data.field_specs[k].tensor_dtype == str else str(data.field_specs[k].tensor_dtype)
            result['__field_spec_tensor_dtype__{}'.format(k)] = tensor_dtype
            result['__field_spec_fill_value__{}'.format(k)] = data.field_specs[k].fill_value
            result['__field_spec_is_sequence__{}'.format(k)] = data.field_specs[k].is_sequence

    np.savez(
        cache_path,
        __num_input_examples__=num_input_examples,
        __has_input_examples__=has_input_examples,
        __num_validation_input_examples__=num_validation_input_examples,
        __has_validation_input_examples__=has_validation_input_examples,
        __num_test_input_examples__=num_test_input_examples,
        __has_test_input_examples__=has_test_input_examples,
        __is_pre_split__=data.is_pre_split,
        __test_proportion__=data.test_proportion,
        __validation_proportion_of_train__=data.validation_proportion_of_train,
        **result)


def _str_to_torch_dtype(s):
    if s in {'str', '<class \'str\'>'}:
        return str
    if s in {'torch.float32', 'torch.float', 'float32', 'float'}:
        return torch.float32
    if s in {'torch.float64', 'torch.double', 'float64', 'double'}:
        return torch.float64
    if s in {'torch.float16', 'torch.half', 'float16', 'half'}:
        return torch.float16
    if s in {'torch.uint8', 'uint8'}:
        return torch.uint8
    if s in {'torch.int8', 'int8'}:
        return torch.int8
    if s in {'torch.int16', 'torch.short', 'int16', 'short'}:
        return torch.int16
    if s in {'torch.int32', 'torch.int', 'int32', 'int'}:
        return torch.int32
    if s in {'torch.int64', 'torch.long', 'int64', 'long'}:
        return torch.long
    raise ValueError('Unknown dtype: {}'.format(s))


def _load_from_cache(cache_path, kwargs, force_cache_miss):
    if force_cache_miss:
        return None

    if not os.path.exists(cache_path):
        return None

    loaded = np.load(cache_path)

    special_keys = [
        '__num_input_examples__',
        '__has_input_examples__',
        '__num_validation_input_examples__',
        '__has_validation_input_examples__',
        '__num_test_input_examples__',
        '__has_test_input_examples__',
        '__lengths__',
        '__is_pre_split__',
        '__test_proportion__',
        '__validation_proportion_of_train__']

    key_prefixes = [
        '__kwarg__',
        '__response_data__',
        '__field_spec_tensor_dtype__',
        '__field_spec_fill_value__',
        '__field_spec_is_sequence__']

    prefix_results = dict()
    example_data = dict()

    for prefix in key_prefixes:
        prefix_results[prefix] = dict()

    for k in loaded.keys():
        is_prefix = False
        for prefix in key_prefixes:
            if k.startswith(prefix):
                prefix_results[prefix][k[len(prefix):]] = loaded[k]
                is_prefix = True
                break
        if not is_prefix:
            if not k.startswith('__'):
                example_data[k] = loaded[k]
            elif k not in special_keys:
                raise ValueError('Unexpected key: {}'.format(k))

    if len(kwargs) != len(prefix_results['__kwarg__']):
        return None

    for k in kwargs:
        if k not in prefix_results['__kwarg__']:
            return None
        if kwargs[k] != prefix_results['__kwarg__'][k].item():
            return None

    tensor_dtypes = prefix_results['__field_spec_tensor_dtype__']
    fill_values = prefix_results['__field_spec_fill_value__']
    is_sequence = prefix_results['__field_spec_is_sequence__']
    assert (len(tensor_dtypes) == len(fill_values))
    assert (len(tensor_dtypes) == len(is_sequence))
    field_specs = dict()
    for k in tensor_dtypes:
        field_specs[k] = FieldSpec(
            fill_value=fill_values[k].item(),
            tensor_dtype=_str_to_torch_dtype(tensor_dtypes[k].item()),
            is_sequence=is_sequence[k].item())

    if len(field_specs) == 0:
        field_specs = None

    all_examples = None
    splits = np.cumsum(loaded['__lengths__'])[:-1]
    for k in example_data:

        if field_specs is not None and k in field_specs and not field_specs[k].is_sequence:
            current = example_data[k]
        else:
            current = np.split(example_data[k], splits)

        if all_examples is None:
            all_examples = [{k: item} for item in current]
        else:
            for idx, item in enumerate(current):
                all_examples[idx][k] = item

    for idx in range(len(all_examples)):
        ex = InputFeatures(**all_examples[idx])
        ex.tokens = [s for s in ex.tokens]  # convert array back to list of tokens
        ex.input_head_tokens = [s for s in ex.input_head_tokens]
        all_examples[idx] = ex

    example_splits = [
        loaded['__num_input_examples__'].item(),
        loaded['__num_input_examples__'].item() + loaded['__num_validation_input_examples__'].item()]

    input_examples = all_examples[:example_splits[0]]
    validation_input_examples = all_examples[example_splits[0]:example_splits[1]]
    test_input_examples = all_examples[example_splits[1]:]

    if not loaded['__has_input_examples__'].item():
        assert(len(input_examples) == 0)
        input_examples = None
    if not loaded['__has_validation_input_examples__'].item():
        assert(len(validation_input_examples) == 0)
        validation_input_examples = None
    if not loaded['__has_test_input_examples__'].item():
        assert(len(test_input_examples) == 0)
        test_input_examples = None

    return RawData(
        input_examples,
        prefix_results['__response_data__'],
        test_input_examples=test_input_examples,
        validation_input_examples=validation_input_examples,
        is_pre_split=loaded['__is_pre_split__'].item(),
        test_proportion=loaded['__test_proportion__'].item(),
        validation_proportion_of_train=loaded['__validation_proportion_of_train__'].item(),
        field_specs=field_specs)


def _populate_default_field_specs(raw_data):
    x, y, z = raw_data.input_examples, raw_data.validation_input_examples, raw_data.test_input_examples
    if x is None:
        x = []
    if y is None:
        y = []
    if z is None:
        z = []
    all_fields = set()
    for ex in itertools.chain(x, y, z):
        all_fields.update([field.name for field in dataclasses.fields(ex)])

    default_field_specs = {
        'unique_id': FieldSpec(tensor_dtype=torch.long, is_sequence=False),
        'tokens': FieldSpec(fill_value='[PAD]', tensor_dtype=str),
        'input_ids': FieldSpec(tensor_dtype=torch.long),
        'input_mask': FieldSpec(tensor_dtype=torch.uint8),
        'input_is_stop': FieldSpec(fill_value=1, tensor_dtype=torch.uint8),
        'input_is_begin_word_pieces': FieldSpec(tensor_dtype=torch.uint8),
        'input_lengths': FieldSpec(tensor_dtype=torch.long),
        'input_probs': FieldSpec(fill_value=-20.),
        'input_head_location': FieldSpec(fill_value=np.nan),
        'input_head_tokens': FieldSpec(fill_value='[PAD]', tensor_dtype=str),
        'input_head_token_ids': FieldSpec(tensor_dtype=torch.long),
        'input_type_ids': FieldSpec(tensor_dtype=torch.long),
        'data_ids': FieldSpec(fill_value=-1, tensor_dtype=torch.long)
    }

    if raw_data.field_specs is None:
        raw_data.field_specs = {}
    for field in all_fields:
        if field not in raw_data.field_specs and field in default_field_specs:
            raw_data.field_specs[field] = default_field_specs[field]


class DataLoader(object):

    geco = 'geco'
    bnc = 'bnc'
    harry_potter = 'harry_potter'
    ucl = 'ucl'
    dundee = 'dundee'
    proto_roles_english_web = 'proto_roles_english_web'
    proto_roles_prop_bank = 'proto_roles_prop_bank'
    natural_stories = 'natural_stories'
    number_dataset = 'number_dataset'

    all_keys = geco, bnc, harry_potter, ucl, dundee, proto_roles_english_web, proto_roles_prop_bank, natural_stories

    def __init__(
            self,
            bert_pre_trained_model_name,
            cache_path,
            geco_path,
            bnc_root,
            harry_potter_path,
            frank_2013_eye_path,
            frank_2015_erp_path,
            dundee_path,
            english_web_universal_dependencies_v_1_2_path,
            english_web_universal_dependencies_v_2_3_path,
            proto_roles_english_web_path,
            ontonotes_path,
            proto_roles_prop_bank_path,
            natural_stories_path,
            number_dataset_path,
            data_key_kwarg_dict=None):
        """
        This object knows how to load data, and stores settings that should be invariant across calls to load
        Args:
            geco_path: The path to the Ghent Eye-tracking Corpus dataset
            bnc_root: The path to the British National Corpus dataset
            harry_potter_path: The path to the Harry Potter dataset
            frank_2015_erp_path: The path to the Frank 2015 dataset
            frank_2013_eye_path: The path to the Frank 2013 dataset
            dundee_path: The path to the Dundee eye-tracking dataset
            english_web_universal_dependencies_v_1_2_path: The path to v1.2 of English Web universal dependencies
                see https://github.com/UniversalDependencies/UD_English-EWT
                and http://universaldependencies.org/#download
                and http://hdl.handle.net/11234/1-1548 (version 1.2 archive)
            english_web_universal_dependencies_v_2_3_path: The path to v2.3 of English Web universal dependencies
                see https://github.com/UniversalDependencies/UD_English-EWT
            proto_roles_english_web_path: The path to the labels for semantic proto roles for v1.2 of
                English Web universal dependencies. See http://decomp.io/data/
            data_key_kwarg_dict: A dictionary keyed by Dataset key (e.g. DataManager.harry_potter), and values
                which are themselves dictionaries. The values are passed as keyword arguments to the underlying load
                functions, e.g.
                kwargs = {}
                if DataManager.harry_potter in self.data_key_kwarg_dict:
                    kwargs = data_key_kwarg_dict[DataManager.harry_potter]
                result = harry_potter_data(self.harry_potter_path, numerical_tokens, start_tokens, **kwargs)
        """
        (self.bert_pre_trained_model_name, self.cache_path, self.geco_path, self.bnc_root,
         self.harry_potter_path, self.frank_2013_eye_path, self.frank_2015_erp_path, self.dundee_path,
         self.english_web_universal_dependencies_v_1_2_path, self.english_web_universal_dependencies_v_2_3_path,
         self.proto_roles_english_web_path, self.ontonotes_path, self.proto_roles_prop_bank_path,
         self.natural_stories_path, self.number_dataset_path, self.data_key_kwarg_dict) = (
            bert_pre_trained_model_name, cache_path, geco_path, bnc_root, harry_potter_path,
            frank_2013_eye_path, frank_2015_erp_path, dundee_path, english_web_universal_dependencies_v_1_2_path,
            english_web_universal_dependencies_v_2_3_path, proto_roles_english_web_path, ontonotes_path,
            proto_roles_prop_bank_path, natural_stories_path, number_dataset_path, data_key_kwarg_dict)

    def make_bert_tokenizer(self):
        return BertTokenizer.from_pretrained(self.bert_pre_trained_model_name, self.cache_path, do_lower_case=True)

    def load(
            self,
            keys,
            data_preparer=None,
            force_cache_miss=False):

        bert_tokenizer = self.make_bert_tokenizer()
        spacy_tokenizer_model = make_tokenizer_model()

        if isinstance(keys, str):
            keys = [keys]

        result = OrderedDict()

        for key in keys:

            cache_path = os.path.join(self.cache_path, '{}.npz'.format(key))

            print('Loading {}...'.format(key), end='', flush=True)

            kwargs = {}
            if self.data_key_kwarg_dict is not None and key in self.data_key_kwarg_dict:
                kwargs = self.data_key_kwarg_dict[key]

            cached = _load_from_cache(cache_path, kwargs, force_cache_miss)

            if cached is None:
                # if key == DataLoader.geco:
                #     result[key] = geco_data(self.geco_path, numerical_tokens, start_tokens, **kwargs)
                # elif key == DataLoader.bnc:
                #     result[key] = bnc_data(
                #         self.bnc_root, numerical_tokens, start_tokens, quick_for_test=quick_for_test, **kwargs)
                if key == DataLoader.ucl:
                    result[key] = ucl_data(
                        spacy_tokenizer_model, bert_tokenizer, self.frank_2013_eye_path,
                        self.frank_2015_erp_path, **kwargs)
                elif key == DataLoader.natural_stories:
                    result[key] = natural_stories_data(
                        spacy_tokenizer_model, bert_tokenizer, self.natural_stories_path, **kwargs)
                elif key == DataLoader.harry_potter:
                    result[key] = harry_potter_data(
                        spacy_tokenizer_model, bert_tokenizer, self.harry_potter_path, **kwargs)
                elif key == DataLoader.number_dataset:
                    result[key] = number_agreement_data(
                        spacy_tokenizer_model, bert_tokenizer, self.number_dataset_path, **kwargs)
                # elif key == DataLoader.dundee:
                #     result[key] = dundee_data(self.dundee_path, numerical_tokens, start_tokens, **kwargs)
                # elif key == DataLoader.proto_roles_english_web:
                #     result[key] = semantic_proto_role_data_english_web(
                #         self.proto_roles_english_web_path, self.english_web_universal_dependencies_v_1_2_path,
                #         numerical_tokens, start_tokens, filter_fn=protocol_v2_1, **kwargs)
                # elif key == DataLoader.proto_roles_prop_bank:
                #     result[key] = semantic_proto_role_data_prop_bank(
                #         self.proto_roles_prop_bank_path,
                #         self.ontonotes_path,
                #         numerical_tokens,
                #         start_tokens,
                #         **kwargs)
                else:
                    raise ValueError('Unrecognized key: {}'.format(key))

                _populate_default_field_specs(result[key])
                _save_to_cache(cache_path, result[key], kwargs)

            else:
                result[key] = cached

            print('done')

        if data_preparer is not None:
            result = data_preparer.prepare(result)

        return result
