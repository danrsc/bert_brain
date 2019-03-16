from collections import OrderedDict
import dataclasses
from typing import Sequence, Union, Optional

import numpy as np

from spacy.language import Language as SpacyLanguage
from pytorch_pretrained_bert import BertTokenizer

from .spacy_token_meta import bert_tokenize_with_spacy_meta
from .input_features import InputFeatures


__all__ = ['CorpusBase', 'CorpusExampleUnifier']


class CorpusExampleUnifier:

    def __init__(self, spacy_tokenize_model: SpacyLanguage, bert_tokenizer: BertTokenizer):
        self.spacy_tokenize_model = spacy_tokenize_model
        self.bert_tokenizer = bert_tokenizer
        self._examples = OrderedDict()

    def add_example(
            self,
            words: Sequence[str],
            data_key: Union[str, Sequence[str]],
            data_ids: Sequence[int],
            is_apply_data_id_to_entire_group: bool = False,
            type_ids: int = 0,
            allow_new_examples: bool = True) -> Optional[InputFeatures]:
        """
        Adds an example for the current data loader to return later. Simplifies the process of merging examples
        across different response measures. For example MEG and fMRI
        Args:
            words: The words in the example
            data_key: A key (or multiple keys) to designate which response data set(s) data_ids references
            data_ids: indices into the response data, one for each key
            is_apply_data_id_to_entire_group: If a word is broken into multiple tokens, generally a single token is
                heuristically chosen as the 'main' token corresponding to that word. The data_id it is assigned is given
                by data offset, while all the tokens that are not the main token in the group are assigned -1. If this
                parameter is set to True, then all of the multiple tokens corresponding to a word are assigned the same
                data_id, and none are set to -1. This can be a better option for fMRI where the predictions are not at
                the word level, but rather at the level of an image containing multiple words.
            type_ids: Used for bert to combine multiple sequences as a single input. Generally this is used for tasks
                like question answering where type_id=0 is the question and type_id=1 is the answer. If not specified,
                type_id=0 is used for every token
            allow_new_examples: If False, then if the example does not already exist in this instance, it will not
                be added. Only new data_ids will be added to existing examples. Returns None when the example does
                not exist.
        Returns:
            The InputFeatures instance associated with the example
        """
        input_features = bert_tokenize_with_spacy_meta(
            self.spacy_tokenize_model, self.bert_tokenizer,
            len(self._examples), words, data_key, data_ids, type_ids, is_apply_data_id_to_entire_group)

        key = tuple(input_features.token_ids)
        if key not in self._examples:
            if allow_new_examples:
                self._examples[key] = input_features
            else:
                return None
        else:
            current = dataclasses.asdict(input_features)
            have = dataclasses.asdict(self._examples[key])
            assert(len(have) == len(current))
            for key in have:
                assert(key in current)
                if key == 'unique_id' or key == 'data_ids':
                    continue
                else:
                    assert np.array_equal(have[key], current[key])
            if isinstance(data_key, str):
                data_key = [data_key]
            for k in data_key:
                self._examples[key].data_ids[k] = input_features.data_ids[k]

        return self._examples[key]

    def iterate_examples(self):
        for k in self._examples:
            yield self._examples[k]


class CorpusBase:
    pass

    def load(self, spacy_tokenizer_model: SpacyLanguage, bert_tokenizer: BertTokenizer):
        example_manager = CorpusExampleUnifier(spacy_tokenizer_model, bert_tokenizer)
        return self._load(example_manager)

    def _load(self, example_manager: CorpusExampleUnifier):
        raise NotImplementedError('{} does not implement _load'.format(type(self)))
