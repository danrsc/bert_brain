from string import punctuation
from dataclasses import dataclass, field
from typing import Union, Sequence, Optional, Tuple

import numpy as np

import spacy
from spacy.language import Language as SpacyLanguage
from spacy.symbols import ORTH
from spacy import parts_of_speech

from transformers import BertTokenizer

from .input_features import InputFeatures

__all__ = ['make_tokenizer_model', 'group_by_cum_lengths', 'get_data_token_index',
           'BertSpacyTokenAligner', 'ChineseCharDetected', 'parts_of_speech_to_ids']

# word’s class was determined from its PoS tag, where nouns, verbs
# (including modal verbs), adjectives, and adverbs were considered
# content words, and all others were function words.

# ADJ	adjective	big, old, green, incomprehensible, first
# ADP	adposition	in, to, during
# ADV	adverb	very, tomorrow, down, where, there
# AUX	auxiliary	is, has (done), will (do), should (do)
# CONJ	conjunction	and, or, but
# CCONJ	coordinating conjunction	and, or, but
# DET	determiner	a, an, the
# INTJ	interjection	psst, ouch, bravo, hello
# NOUN	noun	girl, cat, tree, air, beauty
# NUM	numeral	1, 2017, one, seventy-seven, IV, MMXIV
# PART	particle	's, not,
# PRON	pronoun	I, you, he, she, myself, themselves, somebody
# PROPN	proper noun	Mary, John, London, NATO, HBO
# PUNCT	punctuation	., (, ), ?
# SCONJ	subordinating conjunction	if, while, that
# SYM	symbol	$, %, §, ©, +, −, ×, ÷, =, :), 😝
# VERB	verb	run, runs, running, eat, ate, eating
# X	other
# SPACE	space

content_pos = {'ADJ', 'ADV', 'AUX', 'NOUN', 'PRON', 'PROPN', 'VERB'}


def _is_stop(spacy_token):
    if spacy_token is None:
        return False
    return spacy_token.pos_ not in content_pos


_pos_ids = dict((n, i) for i, n in enumerate([n_ for n_ in parts_of_speech.IDS if n_ != '']))
_pos_ids[''] = np.nan


def parts_of_speech_to_ids():
    result = dict(_pos_ids)
    del result['']
    return result


def _part_of_speech_id(spacy_token):
    # use this to make the values between 0 and n
    if spacy_token is None:
        return np.nan
    return _pos_ids[spacy_token.pos_]


def make_tokenizer_model(model: str = 'en_core_web_md') -> SpacyLanguage:
    model = spacy.load(model)
    # work around for bug in stop words
    for word in model.Defaults.stop_words:
        lex = model.vocab[word]
        lex.is_stop = True

    for w in ('[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]'):
        model.tokenizer.add_special_case(w, [{ORTH: w}])

    return model


def _data_token_better(bert_token_pairs, i, j):
    token_i, _, spacy_i = bert_token_pairs[i]
    token_j, _, spacy_j = bert_token_pairs[j]
    is_continue_i = token_i.startswith('##')
    is_continue_j = token_j.startswith('##')
    if is_continue_i and not is_continue_j:
        return False
    if not is_continue_i and is_continue_j:
        return True
    if _is_stop(spacy_i) and not _is_stop(spacy_j):
        return False
    if not _is_stop(spacy_i) and _is_stop(spacy_j):
        return True
    return len(token_i) > len(token_j)


def get_data_token_index(bert_token_pairs):
    max_i = 0
    for i in range(len(bert_token_pairs)):
        if _data_token_better(bert_token_pairs, i, max_i):
            max_i = i

    return max_i


def group_by_cum_lengths(cum_lengths, tokens):
    group = list()
    current = 0
    for token in tokens:
        while token.idx >= cum_lengths[current]:
            yield group
            group = list()
            current += 1
        group.append(token)

    yield group


def group_word_pieces(bert_tokens):
    group = list()
    for t in bert_tokens:
        s = t
        if isinstance(t, tuple):  # unk
            s = t[0]
        if not s.startswith('##'):
            if len(group) > 0:
                yield group
            group = list()
        group.append(t)
    if len(group) > 0:
        yield group


def align_spacy_meta(spacy_tokens, bert_tokens, word, bert_tokenizer):
    # create character-level is-stop
    char_to_spacy_token = list()

    for idx_token, token in enumerate(spacy_tokens):
        for _ in token.text:
            char_to_spacy_token.append(idx_token)

    if any(t == '[UNK]' for t in bert_tokens):
        resolved_unk_tokens = list()
        basic_tokens = bert_tokenizer.basic_tokenizer.tokenize(word)
        for basic_token in basic_tokens:
            sub_tokens = bert_tokenizer.wordpiece_tokenizer.tokenize(basic_token)
            # it appears that ['UNK'] should only be returned when it is the only thing returned
            if any(t == '[UNK]' for t in sub_tokens):
                assert(len(sub_tokens) == 1)
                resolved_unk_tokens.append((sub_tokens[0], basic_token))
            else:
                resolved_unk_tokens.extend(sub_tokens)
        bert_tokens = resolved_unk_tokens

    char = 0
    result = list()
    for bert_group in group_word_pieces(bert_tokens):
        counts = dict()
        length = 0
        all_punctuation = True
        for t in bert_group:
            if isinstance(t, tuple):
                _, t = t  # [UNK] - use the original for alignment
            if t.startswith('##'):
                t = t[2:]
            length += len(t)
            for c in t:
                if c not in punctuation:
                    all_punctuation = False
                try:
                    spacy_idx = char_to_spacy_token[char]
                except:
                    for x in spacy_tokens:
                        for z in x.text:
                            print(ord(z))
                    print(spacy_tokens[0].is_alpha)
                    print(spacy_tokens[0].is_ascii)
                    print(spacy_tokens[0].is_digit)
                    print(spacy_tokens[0].pos_)
                    print(spacy_tokens[0].tag_)
                    print(spacy_tokens[0].dep_)
                    print(spacy_tokens[0].lang_)
                    print(spacy_tokens, char, bert_group)
                    raise
                if spacy_idx in counts:
                    counts[spacy_idx] += 1
                else:
                    counts[spacy_idx] = 1
                char += 1
        majority_idx = None
        for spacy_idx in counts:
            if majority_idx is None or counts[spacy_idx] > counts[majority_idx]:
                majority_idx = spacy_idx
        spacy_token = spacy_tokens[majority_idx]
        if all_punctuation and not all(c in punctuation for c in spacy_token.text):
            spacy_token = None  # this spacy_token is going to be assigned to a different bert_token
        for idx, t in enumerate(bert_group):
            if isinstance(t, tuple):
                t, _ = t  # [UNK], use the [UNK} now that alignment is complete
            if idx == 0:
                result.append((t, length, spacy_token))
            else:
                result.append((t, 0, None))

    return result


def _get_syntactic_head_group(spacy_token, bert_token_groups):
    if spacy_token is None:
        return None
    if spacy_token.head is None:
        return None
    for idx_group, token_group in enumerate(bert_token_groups):
        for _, _, head in token_group:
            if head is None:
                continue
            if head.idx == spacy_token.head.idx:
                return idx_group
    return None


class ChineseCharDetected(Exception):
    def __init__(self, msg):
        super().__init__(msg)


def is_chinese_character(cp):
    # copied from https://huggingface.co/transformers/_modules/transformers/tokenization_bert.html#BertTokenizer
    # and modified to include Hangul, Hiragana, and Katakana
    if ((0x4E00 <= cp <= 0x9FFF) or
            (0x3400 <= cp <= 0x4DBF) or
            (0x20000 <= cp <= 0x2A6DF) or
            (0x2A700 <= cp <= 0x2B73F) or
            (0x2B740 <= cp <= 0x2B81F) or
            (0x2B820 <= cp <= 0x2CEAF) or
            (0xF900 <= cp <= 0xFAFF) or
            (0x2F800 <= cp <= 0x2FA1F) or
            (0xAC00 <= cp <= 0xD7AF) or  # Hangul
            (0x3040 <= cp <= 0x309F) or  # Hiragana
            (0x30A0 <= cp <= 0x30FF) or  # Katakana
            (0x31F0 <= cp <= 0x31FF) or  # Katakana Phonetic Extensions
            (0x3200 <= cp <= 0x32FF) or  # Enclosed CJK Letters and Months
            (0xFF00 <= cp <= 0xFFEF) or  # Halfwidth and Fullwidth Forms
            (0x1B000 <= cp <= 0x1B0FF) or  # Kana Supplement
            (0x1B130 <= cp <= 0x1B16F)):  # Small Kana Extension
        return True
    return False


class _BertSpacyTokenizeCache:
    def __init__(self):
        self.words = None
        self.bert_token_groups = None
        self.spacy_token_groups = None
        self.bert_token_groups_with_spacy = None

    def match(self, words):
        if self.words is None:
            return False
        if len(words) != len(self.words):
            return False
        for w1, w2 in zip(words, self.words):
            if w1 != w2:
                return False
        return True

    def cache(self, words, bert_token_groups, spacy_token_groups, bert_token_groups_with_spacy):
        self.words = words
        self.bert_token_groups = bert_token_groups
        self.spacy_token_groups = spacy_token_groups
        self.bert_token_groups_with_spacy = bert_token_groups_with_spacy


@dataclass(frozen=True)
class BertSpacyTokenAligner:
    model_tokenizer_name: str = 'bert-base-uncased'
    spacy_language_name: str = 'en_core_web_md'
    cache_path: str = None
    model_tokenizer: BertTokenizer = field(init=False, repr=False, compare=False)
    spacy_language: SpacyLanguage = field(init=False, repr=False, compare=False)
    _align_cache: _BertSpacyTokenizeCache = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            'model_tokenizer',
            BertTokenizer.from_pretrained(self.model_tokenizer_name, cache_dir=self.cache_path, do_lower_case=True))
        object.__setattr__(
            self,
            'spacy_language',
            make_tokenizer_model(self.spacy_language_name))
        object.__setattr__(self, '_align_cache', _BertSpacyTokenizeCache())

    def align(
            self,
            unique_id: int,
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
            multipart_id: Optional[int] = None,
            span_ids: Optional[Sequence[int]] = None,
            is_apply_data_offset_entire_group: bool = False,
            max_sequence_length: Optional[int] = None) -> Tuple[InputFeatures, np.array, int]:
        """
        Uses spacy to get information such as part of speech, probability of word, etc. and aligns the
        tokenization from spacy with the bert tokenization.
        Args:
            unique_id: The unique id for this example
            words: The words in this example. Generally a sentence, but it doesn't have to be.
            sentence_ids: For each word, identifies which sentence the word belongs to. Used to compute
                index_word_in_sentence
            data_key: A key (or multiple keys) to designate which response data set(s) data_ids references
            data_ids: Sequence[Int]. Describes an indices into a separate data array for each word. For example, if the
                first word in words corresponds to fMRI image 17 in a separate data array, and the second word
                corresponds to image 19, then this parameter could start with [17, 19, ...].
            start: Offset where the actual input features should start. It is best to compute spacy meta on full
                sentences, then slice the resulting tokens. start and end are used to slice words, sentence_ids,
                data_key and data_ids
            stop: Exclusive end point for the actual input features. If None, the full length is used
            start_sequence_2: Used for bert to combine 2 sequences as a single input. Generally this is used for tasks
                like question answering where type_id=0 is the question and type_id=1 is the answer. If None, assumes
                the entire input is sequence 1.
            stop_sequence_2: Used for bert to combine 2 sequences as a single input. Generally this is used for tasks
                like question answering where type_id=0 is the question and type_id=1 is the answer. If None, assumes
                the entire input is sequence 1.
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
            is_apply_data_offset_entire_group: If a word is broken into multiple tokens, generally a single token is
                heuristically chosen as the 'main' token corresponding to that word. The data_id it is assigned is given
                by data offset, while all the tokens that are not the main token in the group are assigned -1. If this
                parameter is set to True, then all of the multiple tokens corresponding to a word are assigned the same
                data_id, and none are set to -1. This can be a better option for fMRI where the predictions are not at
                the word level, but rather at the level of an image containing multiple words.
            max_sequence_length: If specified, each sub-sequence will be truncated to this length
        Returns:
            An InputFeatures instance
        """
        if self._align_cache.match(words):
            bert_token_groups = self._align_cache.bert_token_groups
            spacy_token_groups = self._align_cache.spacy_token_groups
            bert_token_groups_with_spacy = self._align_cache.bert_token_groups_with_spacy
        else:
            sent = ''
            cum_lengths = list()

            bert_token_groups = list()
            for w in words:

                for c in w:
                    # noinspection PyProtectedMember
                    if is_chinese_character(ord(c)):
                        raise ChineseCharDetected(w)

                if len(sent) > 0:
                    sent += ' '
                sent += str(w)
                cum_lengths.append(len(sent))
                bert_token_groups.append(self.model_tokenizer.tokenize(w))

            spacy_token_groups = group_by_cum_lengths(cum_lengths, self.spacy_language(sent))

            bert_token_groups_with_spacy = list()
            for spacy_token_group, bert_token_group, word in zip(spacy_token_groups, bert_token_groups, words):
                bert_token_groups_with_spacy.append(
                    align_spacy_meta(spacy_token_group, bert_token_group, word, self.model_tokenizer))

            self._align_cache.cache(words, bert_token_groups, spacy_token_groups, bert_token_groups_with_spacy)

        # bert bert_erp_tokenization does not seem to care whether we do word-by-word or not; it is simple whitespace
        # splitting etc., then sub-word tokens are created from that

        example_tokens = list()
        example_mask = list()
        example_is_stop = list()
        example_pos = list()
        example_pos_id = list()
        example_is_begin_word_pieces = list()
        example_lengths = list()
        example_probs = list()
        example_head_location = list()
        example_token_head = list()
        example_type_ids = list()
        example_data_ids = list()
        example_span_ids = list() if span_ids is not None else None
        example_index_word_in_example = list()
        example_index_token_in_sentence = list()
        included_indices = list()

        def _append_special_token(special_token, index_word_in_example_, index_token_in_sentence_, type_id_):
            example_tokens.append(special_token)
            example_mask.append(1)
            example_is_stop.append(True)
            example_pos.append('')
            example_pos_id.append(np.nan)
            example_is_begin_word_pieces.append(True)
            example_lengths.append(0)
            example_probs.append(-20.)
            example_head_location.append(np.nan)
            example_token_head.append('[PAD]')
            example_type_ids.append(type_id_)
            example_data_ids.append(-1)
            if span_ids is not None:
                example_span_ids.append(0)
            example_index_word_in_example.append(index_word_in_example_)
            example_index_token_in_sentence.append(index_token_in_sentence_)

        type_id = 0

        _append_special_token(
            '[CLS]', index_word_in_example_=0, index_token_in_sentence_=0, type_id_=type_id)

        index_token_in_sentence = 0
        index_word_in_example = 0
        last_sentence_id = None

        if start < 0:
            start = len(words) + start
        if stop is None:
            stop = len(words)
        elif stop < 0:
            stop = len(words) + stop

        sequences = [(start, stop)]

        if start_sequence_2 is not None and start_sequence_2 < 0:
            start_sequence_2 = len(words) + start_sequence_2
        if stop_sequence_2 is not None and stop_sequence_2 < 0:
            stop_sequence_2 = len(words) + stop_sequence_2

        if start_sequence_2 is not None:
            if start_sequence_2 < stop:
                raise ValueError('start_sequence_2 ({}) < stop ({})'.format(start_sequence_2, stop))
            if stop_sequence_2 is None:
                stop_sequence_2 = len(words)
            sequences.append((start_sequence_2, stop_sequence_2))

        if start_sequence_3 is not None and start_sequence_3 < 0:
            start_sequence_3 = len(words) + start_sequence_3
        if stop_sequence_3 is not None and stop_sequence_3 < 0:
            stop_sequence_3 = len(words) + stop_sequence_3

        if stop_sequence_3 is not None:
            if stop_sequence_2 is None or start_sequence_3 < stop_sequence_2:
                raise ValueError(
                    'start_sequence_3 ({}) < stop_sequence_2 ({})'.format(start_sequence_3, stop_sequence_2))
            if stop_sequence_3 is None:
                stop_sequence_3 = len(words)
            sequences.append((start_sequence_3, stop_sequence_3))

        idx_sequence = 0
        current_sequence_length = 0
        true_sequence_length = 0
        max_true_sequence_length = 0
        for idx_group, bert_tokens_with_spacy in enumerate(bert_token_groups_with_spacy):
            if last_sentence_id is None or sentence_ids[idx_group] != last_sentence_id:
                index_token_in_sentence = -1
            last_sentence_id = sentence_ids[idx_group]
            if idx_group >= sequences[idx_sequence][1]:
                if idx_sequence + 1 < len(sequences):
                    idx_sequence += 1
                    current_sequence_length = 0
                    max_true_sequence_length = max(true_sequence_length, max_true_sequence_length)
                    true_sequence_length = 0
                else:
                    break
            if idx_group < sequences[idx_sequence][0]:
                continue
            assert(sequences[idx_sequence][0] <= idx_group < sequences[idx_sequence][1])
            included_indices.append(idx_group)
            index_word_in_example += 1
            idx_data = get_data_token_index(bert_tokens_with_spacy)
            true_sequence_length += len(bert_tokens_with_spacy)
            if max_sequence_length is None \
                    or len(bert_tokens_with_spacy) + current_sequence_length < max_sequence_length:
                for idx_token, (t, length, spacy_token) in enumerate(bert_tokens_with_spacy):
                    index_token_in_sentence += 1
                    current_sequence_length += 1
                    idx_head_group = _get_syntactic_head_group(spacy_token, bert_token_groups_with_spacy)
                    head_token = '[PAD]'
                    head_location = np.nan
                    if idx_head_group is not None:
                        idx_head_data_token = get_data_token_index(bert_token_groups_with_spacy[idx_head_group])
                        head_token = bert_token_groups_with_spacy[idx_head_group][idx_head_data_token][0]
                        head_location = idx_head_group - idx_group
                    example_tokens.append(t)
                    example_mask.append(1)
                    example_is_stop.append(_is_stop(spacy_token))
                    example_pos.append('' if spacy_token is None else spacy_token.pos_)
                    example_pos_id.append(_part_of_speech_id(spacy_token))
                    example_lengths.append(length)
                    example_probs.append(-20. if spacy_token is None else spacy_token.prob)
                    example_head_location.append(head_location)
                    example_token_head.append(head_token)
                    is_continue_word_piece = t.startswith('##')
                    example_is_begin_word_pieces.append(not is_continue_word_piece)
                    example_type_ids.append(type_id)
                    if span_ids is not None:
                        example_span_ids.append(span_ids[idx_group])
                    example_index_word_in_example.append(index_word_in_example)
                    example_index_token_in_sentence.append(index_token_in_sentence)
                    # we follow the BERT paper and always use the first word-piece as the labeled one
                    data_id = -1
                    if data_ids is not None and idx_token == idx_data or is_apply_data_offset_entire_group:
                        data_id = data_ids[idx_group]
                    example_data_ids.append(data_id)
            if idx_group == sequences[idx_sequence][1]:
                _append_special_token('[SEP]', index_word_in_example + 1, index_token_in_sentence + 1, type_id)
                index_word_in_example += 1
                type_id = 1

        if data_key is None:
            data_key = dict()
        if isinstance(data_key, str):
            data_key = [data_key]

        def _readonly(arr):
            arr.setflags(write=False)
            return arr

        example_data_ids = _readonly(np.array(example_data_ids))
        max_true_sequence_length = max(max_true_sequence_length, true_sequence_length)

        input_features = InputFeatures(
            unique_id=unique_id,
            tokens=tuple(example_tokens),
            token_ids=_readonly(np.asarray(self.model_tokenizer.convert_tokens_to_ids(example_tokens))),
            mask=_readonly(np.array(example_mask)),
            is_stop=_readonly(np.array(example_is_stop)),
            part_of_speech=tuple(example_pos),
            part_of_speech_id=_readonly(np.array(example_pos_id)),
            is_begin_word_pieces=_readonly(np.array(example_is_begin_word_pieces)),
            token_lengths=_readonly(np.array(example_lengths)),
            token_probabilities=_readonly(np.array(example_probs)),
            type_ids=_readonly(np.array(example_type_ids)),
            head_location=_readonly(np.array(example_head_location)),
            head_tokens=tuple(example_token_head),
            head_token_ids=_readonly(np.array(self.model_tokenizer.convert_tokens_to_ids(example_token_head))),
            index_word_in_example=_readonly(np.array(example_index_word_in_example)),
            index_token_in_sentence=_readonly(np.array(example_index_token_in_sentence)),
            multipart_id=multipart_id,
            span_ids=_readonly(np.array(example_span_ids)) if example_span_ids is not None else None,
            data_ids=dict((k, example_data_ids) for k in data_key))

        return input_features, np.array(included_indices), max_true_sequence_length
