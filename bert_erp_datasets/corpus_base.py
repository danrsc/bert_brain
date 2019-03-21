from collections import OrderedDict
import dataclasses
from typing import Sequence, Union, Optional, Hashable, Any

import numpy as np

from spacy.language import Language as SpacyLanguage
from pytorch_pretrained_bert import BertTokenizer

from .spacy_token_meta import bert_tokenize_with_spacy_meta
from .input_features import InputFeatures


__all__ = ['CorpusBase', 'CorpusExampleUnifier', 'get_combined_sentence_examples_for_fmri_context']


class CorpusExampleUnifier:

    def __init__(self, spacy_tokenize_model: SpacyLanguage, bert_tokenizer: BertTokenizer):
        self.spacy_tokenize_model = spacy_tokenize_model
        self.bert_tokenizer = bert_tokenizer
        self._examples = OrderedDict()
        self._seen_data_keys = OrderedDict()

    def add_example(
            self,
            example_key: Optional[Hashable],
            words: Sequence[str],
            data_key: Optional[Union[str, Sequence[str]]],
            data_ids: Optional[Sequence[int]],
            is_apply_data_id_to_entire_group: bool = False,
            type_ids: int = 0,
            allow_new_examples: bool = True) -> Optional[InputFeatures]:
        """
        Adds an example for the current data loader to return later. Simplifies the process of merging examples
        across different response measures. For example MEG and fMRI
        Args:
            example_key: For instance, the position of the example within a story. If this is set to None, then the
                tokens will be used as the example_key. However, this may be undesirable since in a given passage,
                sentences, especially short sentences, can be repeated.
            words: The words in the example
            data_key: A key (or multiple keys) to designate which response data set(s) data_ids references
            data_ids: indices into the response data, one for each token
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

        if example_key is None:
            example_key = tuple(input_features.token_ids)

        if example_key not in self._examples:
            if allow_new_examples:
                self._examples[example_key] = input_features
            else:
                return None
        else:
            current = dataclasses.asdict(input_features)
            have = dataclasses.asdict(self._examples[example_key])
            assert(len(have) == len(current))
            for k in have:
                assert(k in current)
                if k == 'unique_id' or k == 'data_ids':
                    continue
                else:
                    # handles NaN, whereas np.array_equal does not
                    np.testing.assert_array_equal(have[k], current[k])
            if data_key is not None:
                if isinstance(data_key, str):
                    data_key = [data_key]
                for k in data_key:
                    self._seen_data_keys[k] = True
                    self._examples[example_key].data_ids[k] = input_features.data_ids[k]

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


class CorpusBase:
    pass

    def load(self, spacy_tokenizer_model: SpacyLanguage, bert_tokenizer: BertTokenizer):
        example_manager = CorpusExampleUnifier(spacy_tokenizer_model, bert_tokenizer)
        return self._load(example_manager)

    def _load(self, example_manager: CorpusExampleUnifier):
        raise NotImplementedError('{} does not implement _load'.format(type(self)))


@dataclasses.dataclass
class _FMRIExample:
    words: Sequence[Any]
    sentence_ids: Sequence[int]
    tr_target: Sequence[Optional[Sequence[int]]]


def get_combined_sentence_examples_for_fmri_context(
        words, word_times, word_sentence_ids, tr_times, duration_tr_features, minimum_duration_required=None,
        tr_offset=0, single_sentence_only=False):

    word_times = np.asarray(word_times)
    if not np.all(np.diff(word_times) >= 0):
        raise ValueError('word_times must be monotonically increasing')
    word_sentence_ids = np.asarray(word_sentence_ids)
    if not np.all(np.diff(word_sentence_ids) >= 0):
        raise ValueError('sentence ids must be monotonically increasing')
    if len(word_times) != len(words):
        raise ValueError('expected one time per word')
    if len(word_sentence_ids) != len(words):
        raise ValueError('expected one sentence is per word')

    word_ids = np.arange(len(words))

    tr_to_sentences = dict()
    word_id_to_trs = dict()
    skipped_trs = set()
    sentence_to_trs = dict()
    for index_tr, tr_time in enumerate(tr_times):

        indicator_words = np.logical_and(word_times >= tr_time - duration_tr_features, word_times < tr_time)

        # nothing is in the window for this tr
        if not np.any(indicator_words):
            skipped_trs.add(index_tr)
            continue

        sentence_ids = np.unique(word_sentence_ids[indicator_words])
        if single_sentence_only:
            sentence_ids = sentence_ids[-1:]
            indicator_sentence_id = word_sentence_ids == sentence_ids[0]
            indicator_words = np.logical_and(indicator_sentence_id, indicator_words)
            if not np.any(indicator_words):
                skipped_trs.add(index_tr)
                continue

        # get the duration from the earliest word in the window to the tr
        # if this is not at least minimum_duration_required then skip the tr
        if minimum_duration_required is not None:
            min_word_time = np.min(word_times[indicator_words])
            if tr_time - min_word_time < minimum_duration_required:
                skipped_trs.add(index_tr)
                continue

        max_word_id = np.max(word_ids[indicator_words])
        if max_word_id not in word_id_to_trs:
            word_id_to_trs[max_word_id] = list()
        word_id_to_trs[max_word_id].append(index_tr)

        tr_to_sentences[index_tr] = sentence_ids
        for sentence_id in sentence_ids:
            if sentence_id not in sentence_to_trs:
                sentence_to_trs[sentence_id] = list()
            sentence_to_trs[sentence_id].append(index_tr)

    result = list()
    output_trs = set()
    for tr in tr_to_sentences:
        sentences = tr_to_sentences[tr]
        overlapping_trs = set(sentence_to_trs[sentences[0]])
        sentences = set(sentences)
        is_owner = True
        for tr2 in overlapping_trs:
            if tr2 == tr:
                continue
            tr2_sentences = set(tr_to_sentences[tr2])
            if sentences.issubset(tr2_sentences):
                # same set, ownership goes to the first tr
                if len(sentences) == len(tr2_sentences):
                    if tr2 < tr:
                        is_owner = False
                        break
                else:
                    is_owner = False
        if not is_owner:
            continue
        sentences = sorted(sentences)
        example_words = list()
        for sentence in sentences:
            example_words.extend(word_ids[word_sentence_ids == sentence])

        tr_targets = [word_id_to_trs[w] if w in word_id_to_trs else None for w in example_words]

        current_trs = set()
        for target in tr_targets:
            if target is not None:
                assert(all([t not in current_trs for t in target]))
                current_trs.update(target)

        output_trs.update(current_trs)
        if tr_offset != 0:
            for idx in range(len(tr_targets)):
                if tr_targets[idx] is not None:
                    tr_targets[idx] = [t + tr_offset for t in tr_targets[idx]]
        result.append(_FMRIExample(
            [words[i] for i in example_words],
            [word_sentence_ids[i] for i in example_words],
            tr_targets))

    assert(all([t in output_trs or t in skipped_trs for t in range(len(tr_times))]))

    return result
