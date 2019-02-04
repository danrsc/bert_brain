from string import punctuation
import numpy as np
import spacy
from spacy.symbols import ORTH

from .input_features import InputFeatures

__all__ = ['tokenize_single', 'make_tokenizer_model', 'group_by_cum_lengths', 'get_data_token_index',
           'bert_tokenize_with_spacy_meta']

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
# X	other	sfpksdpsxmsa
# SPACE	space

content_pos = {'ADJ', 'ADV', 'AUX', 'NOUN', 'PRON', 'PROPN', 'VERB'}


def _iterate_tokens(tokens):
    # a little generator utility to filter whitespace tokens and convert to fields
    for idx in range(len(tokens)):
        token = tokens[idx]

        # ignore the is_stop from spacy; use pos definition
        is_stop = token.pos_ not in content_pos

        # yield an additional token to capture shape
        if token.shape_.isupper() and len(token.shape_) > 2:
            yield 't_up', token.idx, True, -20.0
        yield token.text.lower(), token.idx, is_stop, token.prob


def tokenize_single(s, model):
    tokens = model(s)
    return list(_iterate_tokens(tokens))


def make_tokenizer_model(model='en_core_web_md'):
    model = spacy.load(model)
    # work around for bug in stop words
    for word in model.Defaults.stop_words:
        lex = model.vocab[word]
        lex.is_stop = True

    for w in ('<eos>', '<bos>', '<unk>'):
        model.tokenizer.add_special_case(w, [{ORTH: w}])

    return model


def _data_token_better(bert_token_pairs, i, j):
    token_i, is_stop_i = bert_token_pairs[i]
    token_j, is_stop_j = bert_token_pairs[j]
    is_continue_i = token_i.startswith('##')
    is_continue_j = token_j.startswith('##')
    if is_continue_i and not is_continue_j:
        return False
    if not is_continue_i and is_continue_j:
        return True
    if is_stop_i and not is_stop_j:
        return False
    if not is_stop_i and is_stop_j:
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
    for token, idx, is_stop, prob in tokens:
        while idx >= cum_lengths[current]:
            yield group
            group = list()
            current += 1
        group.append((token, idx, is_stop, prob))

    yield group


def align_spacy_stop(spacy_tokens, bert_tokens):
    # create character-level is-stop
    char_is_stop = list()
    for t, idx, is_stop, prob in spacy_tokens:
        for c in t:
            char_is_stop.append(is_stop)
    idx = 0
    bert_stops = list()
    for t in bert_tokens:
        if t.startswith('##'):
            t = t[2:]
        all_stop = True
        for c in t:
            current = char_is_stop[idx]
            idx += 1
            if c not in punctuation and not current:
                all_stop = False
        bert_stops.append(all_stop)
    return list(zip(bert_tokens, bert_stops))


def bert_tokenize_with_spacy_meta(
        spacy_model, bert_tokenizer, max_sequence_length, unique_id, words, data_offset, type_id=0):

    sent = ''
    cum_lengths = list()

    bert_token_groups = list()
    for w in words:

        if len(sent) > 0:
            sent += ' '
        sent += str(w)
        cum_lengths.append(len(sent))
        bert_token_groups.append(bert_tokenizer.tokenize(w))

    spacy_token_groups = group_by_cum_lengths(cum_lengths, tokenize_single(sent, spacy_model))

    # bert bert_erp_tokenization does not seem to care whether we do word-by-word or not; it is simple whitespace
    # splitting etc., then sub-word tokens are created from that

    example_tokens = list()
    example_mask = list()
    example_is_stop = list()
    example_is_begin_word_pieces = list()
    example_type_ids = list()
    example_data_ids = list()

    def _append_special_token(special_token):
        example_tokens.append(special_token)
        example_mask.append(1)
        example_is_stop.append(1)
        example_is_begin_word_pieces.append(1)
        example_type_ids.append(type_id)
        example_data_ids.append(-1)

    _append_special_token('[CLS]')

    for idx_group, (spacy_token_group, bert_token_group) in enumerate(zip(spacy_token_groups, bert_token_groups)):
        # for now, we are not worrying about prob; just the stop words
        bert_tokens_with_stop = align_spacy_stop(spacy_token_group, bert_token_group)
        idx_data = get_data_token_index(bert_tokens_with_stop)
        for idx_token, (t, is_stop) in enumerate(bert_tokens_with_stop):
            example_tokens.append(t)
            example_mask.append(1)
            example_is_stop.append(1 if is_stop else 0)
            is_continue_word_piece = t.startswith('##')
            example_is_begin_word_pieces.append(0 if is_continue_word_piece else 1)
            example_type_ids.append(type_id)
            # we follow the BERT paper and always use the first word-piece as the labeled one
            example_data_ids.append(data_offset + idx_group if idx_token == idx_data else -1)

    _append_special_token('[SEP]')

    if len(example_tokens) > max_sequence_length:
        example_tokens = example_tokens[:max_sequence_length]
        example_mask = example_mask[:max_sequence_length]
        example_is_stop = example_is_stop[:max_sequence_length]
        example_is_begin_word_pieces = example_is_begin_word_pieces[:max_sequence_length]
        example_type_ids = example_type_ids[:max_sequence_length]
        example_data_ids = example_data_ids[:max_sequence_length]
        raise ValueError('Raise max_sequence_length; not currently supporting multiple examples per document')

    return InputFeatures(
        unique_id=unique_id,
        tokens=np.array(example_tokens),
        input_ids=np.asarray(bert_tokenizer.convert_tokens_to_ids(example_tokens)),
        input_mask=np.array(example_mask),
        input_is_stop=np.array(example_is_stop),
        input_is_begin_word_pieces=np.array(example_is_begin_word_pieces),
        input_type_ids=np.array(example_type_ids),
        data_ids=np.array(example_data_ids))
