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


def bert_tokenize_with_spacy_meta(
        spacy_model, bert_tokenizer, max_sequence_length, unique_id, words, data_offset, type_id=0):

    sent = ''
    cum_lengths = list()

    for w in words:

        if len(sent) > 0:
            sent += ' '
        sent += str(w)
        cum_lengths.append(len(sent))

    word_token_groups = group_by_cum_lengths(cum_lengths, tokenize_single(sent, spacy_model))
    # bert tokenization does not seem to care whether we do word-by-word or not; it is simple whitespace
    # splitting etc., then sub-word tokens are created from that

    example_tokens = list()
    example_mask = list()
    example_is_stop = list()
    example_type_ids = list()
    example_data_ids = list()

    for special_token in ['[CLS]', '[SEP]']:
        example_tokens.append(special_token)
        example_mask.append(1)
        example_is_stop.append(1)
        example_type_ids.append(type_id)
        example_data_ids.append(-1)

    for idx_group, token_group in enumerate(word_token_groups):
        group_bert_tokenized = list()
        for token, idx, is_stop, prob in token_group:
            # for now, we are not worrying about prob; just the stop words
            for bert_token in bert_tokenizer.tokenize(token):
                group_bert_tokenized.append((bert_token, is_stop))

        idx_data = get_data_token_index(group_bert_tokenized) if len(group_bert_tokenized) > 1 else 0

        for idx_token, (t, is_stop) in enumerate(group_bert_tokenized):
            example_tokens.append(t)
            example_mask.append(1)
            example_is_stop.append(is_stop)
            example_type_ids.append(type_id)
            example_data_ids.append(data_offset + idx_group if idx_token == idx_data else -1)

    if len(example_tokens) > max_sequence_length:
        example_tokens = example_tokens[:max_sequence_length]
        example_mask = example_mask[:max_sequence_length]
        example_is_stop = example_is_stop[:max_sequence_length]
        example_type_ids = example_type_ids[:max_sequence_length]
        example_data_ids = example_data_ids[:max_sequence_length]

    return InputFeatures(
        unique_id=unique_id,
        tokens=example_tokens,
        input_ids=bert_tokenizer.convert_tokens_to_ids(example_tokens),
        input_mask=example_mask,
        input_is_stop=example_is_stop,
        input_type_ids=example_type_ids,
        data_ids=example_data_ids)
