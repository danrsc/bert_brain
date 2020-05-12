import os
from collections import OrderedDict
import itertools
import dataclasses
from typing import Sequence, Union, Optional, Hashable, Tuple, Any

import numpy as np
import torch

from .spacy_token_meta import BertSpacyTokenAligner
from .input_features import InputFeatures, FieldSpec, RawData


__all__ = ['CorpusBase', 'CorpusExampleUnifier', 'path_attribute_field']


class CorpusExampleUnifier:

    def __init__(
            self,
            bert_spacy_token_aligner: BertSpacyTokenAligner,
            max_sequence_length: Optional[int] = None):
        self._examples = OrderedDict()
        self._seen_data_keys = OrderedDict()
        self.max_sequence_length = max_sequence_length
        self._true_max_sequence_length = 0
        self.token_aligner = bert_spacy_token_aligner

    @property
    def true_max_sequence_length(self):
        return self._true_max_sequence_length

    def add_example(
            self,
            example_key: Optional[Hashable],
            words: Sequence[str],
            sentence_ids: Sequence[int],
            data_key: Optional[Union[str, Sequence[str]]],
            data_ids: Optional[Sequence[int]],
            start: int = 0,
            stop: Optional[int] = None,
            start_sequence_2: Optional[int] = None,
            stop_sequence_2: Optional[int] = None,
            start_sequence_3: Optional[int] = None,
            stop_sequence_3: Optional[int] = None,
            is_apply_data_id_to_entire_group: bool = False,
            multipart_id: Optional[int] = None,
            span_ids: Optional[Sequence[int]] = None,
            allow_new_examples: bool = True,
            return_included_indices: bool = False,
            allow_duplicates: bool = True,
            auto_high_frequency_on_collision: bool = False) -> Union[
                Optional[InputFeatures], Optional[Tuple[InputFeatures, np.array]]]:
        """
        Adds an example for the current data loader to return later. Simplifies the process of merging examples
        across different response measures. For example MEG and fMRI
        Args:
            example_key: For instance, the position of the example within a story. If this is set to None, then the
                tokens will be used as the example_key. However, this may be undesirable since in a given passage,
                sentences, especially short sentences, can be repeated.
            words: The words in the example
            sentence_ids: For each word, identifies which sentence the word belongs to. Used to compute
                index_of_word_in_sentence in the resulting InputFeatures
            data_key: A key (or multiple keys) to designate which response data set(s) data_ids references
            data_ids: indices into the response data, one for each token
            start: Offset where the actual input features should start. It is best to compute spacy meta on full
                sentences, then slice the resulting tokens. start and stop are used to slice words, sentence_ids,
                data_ids and type_ids
            stop: Exclusive end point for the actual input features. If None, the full length is used
            is_apply_data_id_to_entire_group: If a word is broken into multiple tokens, generally a single token is
                heuristically chosen as the 'main' token corresponding to that word. The data_id it is assigned is given
                by data offset, while all the tokens that are not the main token in the group are assigned -1. If this
                parameter is set to True, then all of the multiple tokens corresponding to a word are assigned the same
                data_id, and none are set to -1. This can be a better option for fMRI where the predictions are not at
                the word level, but rather at the level of an image containing multiple words.
            start_sequence_2: Used for bert to combine multiple sequences as a single input. Generally this is used for
                tasks like question answering where type_id=0 is the question and type_id=1 is the answer.
                If not specified, type_id=0 is used for every token
            stop_sequence_2: Used for bert to combine multiple sequences as a single input. Generally this is used for
                tasks like question answering where type_id=0 is the question and type_id=1 is the answer.
            start_sequence_3: Used for bert to combine 3 sequences as a single input. Generally this is used for tasks
                like question answering with a context. type_id=0 is the context and type_id=1 is the question and
                answer
            stop_sequence_3: Used for bert to combine 3 sequences as a single input. Generally this is used for tasks
                like question answering with a context. type_id=0 is the context and type_id=1 is the question and
                answer
            multipart_id: Used to express that this example needs to be in the same batch as other examples sharing the
                same multipart_id to be evaluated
            span_ids: Bit-encoded span identifiers which indicate which spans each word belongs to when spans are
                labeled in the input. If not given, no span ids will be set on the returned InputFeatures instance.
            allow_new_examples: If False, then if the example does not already exist in this instance, it will not
                be added. Only new data_ids will be added to existing examples. Returns None when the example does
                not exist.
            return_included_indices: If True, then the indices into words determined by start, stop,
                start_sequence_2, etc. are returned to the caller
            allow_duplicates: If False and an example key already exists in the Corpus, then None is returned rather
                than the existing example and the data_ids are not merged
            auto_high_frequency_on_collision: If True and two examples have the same example keys but different
                token_probabilities (because the case of the input words differs between the examples), then the
                token_probabilities is set to
                np.maximum(current_example.token_probabilities, new_example.token_probabilities)
        Returns:
            The InputFeatures instance associated with the example
        """
        input_features, included_indices, true_sequence_length = self.token_aligner.align(
            len(self._examples), words, sentence_ids, data_key, data_ids,
            start, stop,
            start_sequence_2, stop_sequence_2,
            start_sequence_3, stop_sequence_3,
            multipart_id,
            span_ids,
            is_apply_data_id_to_entire_group,
            max_sequence_length=self.max_sequence_length)

        if example_key is None:
            example_key = tuple(input_features.token_ids)

        if example_key not in self._examples:
            if allow_new_examples:
                self._examples[example_key] = input_features
                self._true_max_sequence_length = max(self._true_max_sequence_length, true_sequence_length)
            else:
                return None
        else:
            check_collision = True
            if (auto_high_frequency_on_collision
                    and
                    len(input_features.token_probabilities) == len(self._examples[example_key].token_probabilities)):
                check_collision = False
                if np.sum(input_features.token_probabilities) > np.sum(self._examples[example_key].token_probabilities):
                    # overwrite the token-dependent input_features with the higher probability example
                    current = dataclasses.asdict(input_features)
                    del current['unique_id']
                    del current['multipart_id']
                    del current['data_ids']
                    self._examples[example_key] = dataclasses.replace(self._examples[example_key], **current)

            if check_collision:
                current = dataclasses.asdict(input_features)
                have = dataclasses.asdict(self._examples[example_key])
                assert(len(have) == len(current))
                for k in have:
                    assert(k in current)
                    if k == 'unique_id' or k == 'data_ids' or k == 'multipart_id':
                        continue
                    else:
                        # handles NaN and various typing issues,
                        # whereas np.array_equal does not
                        np.testing.assert_array_equal(
                            have[k], current[k], 'mismatch between duplicate example keys. {}'.format(k))
            if not allow_duplicates:
                return None

        if data_key is not None:
            if isinstance(data_key, str):
                data_key = [data_key]
            for k in data_key:
                self._seen_data_keys[k] = True
                self._examples[example_key].data_ids[k] = input_features.data_ids[k]

        if return_included_indices:
            return self._examples[example_key], included_indices
        return self._examples[example_key]

    def iterate_examples(self, fill_data_keys=False):
        for k in self._examples:
            if fill_data_keys:
                for data_key in self._seen_data_keys:
                    if data_key not in self._examples[k].data_ids:
                        self._examples[k].data_ids[data_key] = -1 * np.ones(
                            len(self._examples[k].token_ids), dtype=np.int64)
            yield self._examples[k]

    def remove_data_keys(self, data_keys):
        if isinstance(data_keys, str):
            data_keys = [data_keys]
        for ex in self.iterate_examples():
            for data_key in data_keys:
                if data_key in ex.data_ids:
                    del ex.data_ids[data_key]
        for data_key in data_keys:
            if data_key in self._seen_data_keys:
                del self._seen_data_keys[data_key]

    def __len__(self):
        return len(self._examples)


def path_attribute_field(attribute: str) -> dataclasses.Field:
    return dataclasses.field(default=None, metadata=dict(path_attribute=attribute))


@dataclasses.dataclass(frozen=True)
class CorpusBase:
    run_info: Any = None
    cache_base_path: str = None
    index_run: dataclasses.InitVar[Optional[int]] = None

    def __post_init__(self, index_run: Optional[int]):
        if index_run is not None:
            object.__setattr__(self, 'run_info', self._run_info(index_run))

    @property
    def corpus_key(self):
        return type(self).__name__

    @staticmethod
    def replace_paths(corpus: 'CorpusBase', path_object, **additional_changes) -> 'CorpusBase':
        path_kwargs = dict(additional_changes)
        for field in dataclasses.fields(corpus):
            val = getattr(corpus, field.name)
            if val is None:
                if 'path_attribute' in field.metadata:
                    path_attribute = field.metadata['path_attribute']
                    if not hasattr(path_object, path_attribute):
                        raise ValueError('path_object does not have an attribute called {}'.format(path_attribute))
                    path_kwargs[field.name] = getattr(path_object, path_attribute)
                elif field.name == 'cache_base_path':
                    path_kwargs[field.name] = os.path.join(path_object.cache_path, corpus.corpus_key)
        return dataclasses.replace(corpus, **path_kwargs)

    @staticmethod
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
            'token_ids': FieldSpec(tensor_dtype=torch.long),
            'mask': FieldSpec(tensor_dtype=torch.uint8),
            'is_stop': FieldSpec(fill_value=True, tensor_dtype=torch.bool),
            'part_of_speech': FieldSpec(fill_value='', tensor_dtype=str),
            'part_of_speech_id': FieldSpec(tensor_dtype=torch.float),
            'is_begin_word_pieces': FieldSpec(tensor_dtype=torch.bool),
            'token_lengths': FieldSpec(tensor_dtype=torch.long),
            'token_probabilities': FieldSpec(fill_value=-20.),
            'head_location': FieldSpec(fill_value=np.nan),
            'head_tokens': FieldSpec(fill_value='[PAD]', tensor_dtype=str),
            'head_token_ids': FieldSpec(tensor_dtype=torch.long),
            'type_ids': FieldSpec(tensor_dtype=torch.long),
            'data_ids': FieldSpec(fill_value=-1, tensor_dtype=torch.long),
            'span_ids': FieldSpec(fill_value=0, tensor_dtype=torch.long),
            'index_word_in_example': FieldSpec(fill_value=-1, tensor_dtype=torch.long),
            'index_token_in_sentence': FieldSpec(fill_value=0, tensor_dtype=torch.long),
            'multipart_id': FieldSpec(tensor_dtype=torch.long, is_sequence=False)
        }

        if raw_data.field_specs is None:
            raw_data.field_specs = {}
        for field in all_fields:
            if field not in raw_data.field_specs and field in default_field_specs:
                raw_data.field_specs[field] = default_field_specs[field]

    def load(
            self,
            bert_spacy_token_aligner: BertSpacyTokenAligner,
            max_sequence_length: Optional[int] = None,
            use_meta_train: bool = False) -> Tuple[RawData, int]:
        example_manager = CorpusExampleUnifier(bert_spacy_token_aligner, max_sequence_length)
        result = self._load(example_manager, use_meta_train)
        CorpusBase._populate_default_field_specs(result)
        return result, example_manager.true_max_sequence_length

    def _run_info(self, index_run):
        return -1

    def _load(self, example_manager: CorpusExampleUnifier, use_meta_train: bool) -> RawData:
        raise NotImplementedError('{} does not implement _load'.format(type(self)))
