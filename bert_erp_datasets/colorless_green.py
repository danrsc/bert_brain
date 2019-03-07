import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from bert_erp_common import zip_equal
from bert_erp_tokenization import bert_tokenize_with_spacy_meta, RawData, FieldSpec
from syntactic_dependency import preprocess_english_morphology, collect_paradigms, extract_dependency_patterns, \
    generate_morph_pattern_test, DependencyTree, universal_dependency_reader, make_token_to_paradigms, \
    make_ltm_to_word, GeneratedExample


__all__ = ['colorless_green_agreement_data', 'linzen_agreement_data']


def _iterate_delimited(path, field_delimiter='\t', pattern_field_delimiter='!', pattern_context_delimiter='_'):
    with open(path, 'rt') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#') or len(line) == 0:
                continue
            yield GeneratedExample.from_delimited(
                line, field_delimiter, pattern_field_delimiter, pattern_context_delimiter)


@dataclass
class _LinzenExample:
    words: Tuple[str, ...]
    index_target: int
    correct_form: str
    incorrect_form: str
    num_attractors: int

    @property
    def agreement_tuple(self):
        return self.words, self.correct_form, self.incorrect_form, self.index_target


def _iterate_linzen(directory_path):
    with open(os.path.join(directory_path, 'subj_agr_filtered.text'), 'rt') as sentence_file:
        with open(os.path.join(directory_path, 'subj_agr_filtered.gold'), 'rt') as gold_file:
            for sentence, gold in zip_equal(sentence_file, gold_file):
                index_target, correct_form, incorrect_form, num_attractors = gold.split('\t')
                yield _LinzenExample(
                    sentence.split()[:-1],  # remove <eos>
                    int(index_target),
                    correct_form,
                    incorrect_form,
                    int(num_attractors))


def generate_examples(english_web_path, bert_tokenizer):

    conll_reader = universal_dependency_reader

    if isinstance(english_web_path, str):
        english_web_path = [english_web_path]

    paradigms = collect_paradigms(english_web_path, morphology_preprocess_fn=preprocess_english_morphology)

    trees = [
        DependencyTree.from_conll_rows(sentence_rows, conll_reader.root_index, conll_reader.offset, text)
        for sentence_rows, text in conll_reader.iterate_sentences_chain_streams(
            english_web_path,
            morphology_preprocess_fn=preprocess_english_morphology)]

    syntax_patterns = extract_dependency_patterns(trees, freq_threshold=5, feature_keys={'Number'})

    paradigms = make_token_to_paradigms(paradigms)

    ltm_paradigms = make_ltm_to_word(paradigms)

    examples = list()
    for pattern in syntax_patterns:
        examples.extend(generate_morph_pattern_test(trees, pattern, ltm_paradigms, paradigms, bert_tokenizer))

    return examples


def _agreement_data(spacy_tokenize_model, bert_tokenizer, examples):
    class_correct = 1
    class_incorrect = 0
    classes = list()
    input_examples = list()

    for example in examples:
        words, correct_form, incorrect_form, index_target = example.agreement_tuple
        words = list(words)

        # the generated example actually doesn't use the test item (the form field); it is a different random word
        # until we put the test item in there
        words[index_target] = correct_form

        input_example = bert_tokenize_with_spacy_meta(
            spacy_tokenize_model, bert_tokenizer, len(input_examples), words,
            data_offset=lambda idx_word: len(input_examples) if idx_word == index_target else -1)
        classes.append(class_correct)
        input_examples.append(input_example)

        # switch to the wrong number-agreement
        words[index_target] = incorrect_form

        input_example = bert_tokenize_with_spacy_meta(
            spacy_tokenize_model, bert_tokenizer, len(input_examples), words,
            data_offset=lambda idx_word: len(input_examples) if idx_word == index_target else -1)
        classes.append(class_incorrect)
        input_examples.append(input_example)

    return input_examples, classes


def colorless_green_agreement_data(spacy_tokenize_model, bert_tokenizer, path):

    english_web_paths = [
        os.path.join(path, 'en_ewt-ud-train.conllu'),
        os.path.join(path, 'en_ewt-ud-dev.conllu'),
        os.path.join(path, 'en_ewt-ud-test.conllu')]

    examples, classes = _agreement_data(
        spacy_tokenize_model, bert_tokenizer, generate_examples(english_web_paths, bert_tokenizer))

    classes = {'colorless': np.array(classes, dtype=np.float64)}

    return RawData(
        examples, classes,
        validation_proportion_of_train=0.1, field_specs={'colorless': FieldSpec(is_sequence=False)})


def linzen_agreement_data(spacy_tokenize_model, bert_tokenizer, path):

    examples, classes = _agreement_data(spacy_tokenize_model, bert_tokenizer, _iterate_linzen(path))

    classes = {'linzen_agree': np.array(classes, dtype=np.float)}
    return RawData(
        examples, classes,
        validation_proportion_of_train=0.1, field_specs={'linzen_agree': FieldSpec(is_sequence=False)})
