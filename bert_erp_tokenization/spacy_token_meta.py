from string import punctuation
import numpy as np
import spacy
from spacy.symbols import ORTH

from .input_features import InputFeatures

__all__ = ['make_tokenizer_model', 'group_by_cum_lengths', 'get_data_token_index',
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


def _is_stop(spacy_token):
    if spacy_token is None:
        return False
    return spacy_token.pos_ not in content_pos


def make_tokenizer_model(model='en_core_web_md'):
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
                spacy_idx = char_to_spacy_token[char]
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


def bert_tokenize_with_spacy_meta(spacy_model, bert_tokenizer, unique_id, words, data_offset, type_id=0):

    sent = ''
    cum_lengths = list()

    bert_token_groups = list()
    for w in words:

        if len(sent) > 0:
            sent += ' '
        sent += str(w)
        cum_lengths.append(len(sent))
        bert_token_groups.append(bert_tokenizer.tokenize(w))

    spacy_token_groups = group_by_cum_lengths(cum_lengths, spacy_model(sent))

    # bert bert_erp_tokenization does not seem to care whether we do word-by-word or not; it is simple whitespace
    # splitting etc., then sub-word tokens are created from that

    example_tokens = list()
    example_mask = list()
    example_is_stop = list()
    example_is_begin_word_pieces = list()
    example_lengths = list()
    example_probs = list()
    example_head_location = list()
    example_token_head = list()
    example_type_ids = list()
    example_data_ids = list()

    def _append_special_token(special_token):
        example_tokens.append(special_token)
        example_mask.append(1)
        example_is_stop.append(1)
        example_is_begin_word_pieces.append(1)
        example_lengths.append(0)
        example_probs.append(-20.)
        example_head_location.append(np.nan)
        example_token_head.append('[PAD]')
        example_type_ids.append(type_id)
        example_data_ids.append(-1)

    _append_special_token('[CLS]')

    bert_token_groups_with_spacy = list()
    for spacy_token_group, bert_token_group, word in zip(spacy_token_groups, bert_token_groups, words):
        bert_token_groups_with_spacy.append(align_spacy_meta(spacy_token_group, bert_token_group, word, bert_tokenizer))

    for idx_group, bert_tokens_with_spacy in enumerate(bert_token_groups_with_spacy):
        idx_data = get_data_token_index(bert_tokens_with_spacy)
        for idx_token, (t, length, spacy_token) in enumerate(bert_tokens_with_spacy):
            idx_head_group = _get_syntactic_head_group(spacy_token, bert_token_groups_with_spacy)
            head_token = '[PAD]'
            head_location = np.nan
            if idx_head_group is not None:
                idx_head_data_token = get_data_token_index(bert_token_groups_with_spacy[idx_head_group])
                head_token = bert_token_groups_with_spacy[idx_head_group][idx_head_data_token][0]
                head_location = idx_head_group - idx_group
            example_tokens.append(t)
            example_mask.append(1)
            example_is_stop.append(1 if _is_stop(spacy_token) else 0)
            example_lengths.append(length)
            example_probs.append(-20. if spacy_token is None else spacy_token.prob)
            example_head_location.append(head_location)
            example_token_head.append(head_token)
            is_continue_word_piece = t.startswith('##')
            example_is_begin_word_pieces.append(0 if is_continue_word_piece else 1)
            example_type_ids.append(type_id)
            # we follow the BERT paper and always use the first word-piece as the labeled one
            data_id = -1
            if idx_token == idx_data:
                if callable(data_offset):
                    data_id = data_offset(idx_group)
                elif data_offset >= 0:
                    data_id = data_offset + idx_group
            example_data_ids.append(data_id)

    _append_special_token('[SEP]')

    return InputFeatures(
        unique_id=unique_id,
        tokens=example_tokens,
        input_ids=np.asarray(bert_tokenizer.convert_tokens_to_ids(example_tokens)),
        input_mask=np.array(example_mask),
        input_is_stop=np.array(example_is_stop),
        input_is_begin_word_pieces=np.array(example_is_begin_word_pieces),
        input_lengths=np.array(example_lengths),
        input_probs=np.array(example_probs),
        input_head_location=np.array(example_head_location),
        input_head_tokens=example_token_head,
        input_head_token_ids=np.array(bert_tokenizer.convert_tokens_to_ids(example_token_head)),
        input_type_ids=np.array(example_type_ids),
        data_ids=np.array(example_data_ids))
