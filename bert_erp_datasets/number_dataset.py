from collections import OrderedDict
from dataclasses import dataclass
import dataclasses
from typing import Tuple

import numpy as np

from bert_erp_tokenization import bert_tokenize_with_spacy_meta, RawData


__all__ = ['SyntaxPattern', 'GeneratedExample', 'number_agreement_data']


@dataclass
class SyntaxPattern:
    arc_direction: str
    context: Tuple[str, ...]
    left_value_1: str
    left_value_2: str

    def delimited(self, field_delimiter='!', context_delimiter='_'):
        d = dataclasses.asdict(self, dict_factory=OrderedDict)
        result = list()
        for field in d:
            if field == 'context':
                result.append(context_delimiter.join(d[field]))
            else:
                result.append('{}'.format(d[field]))
        return field_delimiter.join(result)

    @classmethod
    def from_delimited(cls, delimited, field_delimiter='!', context_delimiter='_'):
        fields = dataclasses.fields(cls)
        values = delimited.split(field_delimiter)
        if len(values) != len(fields):
            raise ValueError('Number of fields in input ({}) does not match number of fields in {} ({})'.format(
                len(values), cls, len(fields)))
        d = dict()
        for field, str_value in zip(fields, values):
            if field.name == 'context':
                d[field.name] = str_value.split(context_delimiter)
            else:
                d[field.name] = field.type(str_value)
        return cls(**d)


@dataclass
class GeneratedExample:
    pattern: SyntaxPattern
    construction_id: int
    sentence_id: int
    right_index: int
    right_pos: str
    right_morph: str
    form: str
    number: str
    alternate_form: str
    lemma: str
    left_index: int
    left_pos: str
    prefix: str
    generated_context: str

    def delimited(self, field_delimiter='\t', pattern_field_delimiter='!', pattern_context_delimiter='_'):
        d = dataclasses.asdict(self, dict_factory=OrderedDict)
        result = list()
        for field in d:
            if field == 'pattern':
                result.append(d[field].delimited(pattern_field_delimiter, pattern_context_delimiter))
            else:
                result.append('{}'.format(d[field]))
        return field_delimiter.join(result)

    @classmethod
    def from_delimited(
            cls, delimited, field_delimiter='\t', pattern_field_delimiter='!', pattern_context_delimiter='_'):
        fields = dataclasses.fields(cls)
        values = delimited.split(field_delimiter)
        if len(values) != len(fields):
            raise ValueError('Number of fields in input ({}) does not match number of fields in {} ({})'.format(
                len(values), cls, len(fields)))
        d = dict()
        for field, str_value in zip(fields, values):
            if field.name == 'pattern':
                d[field.name] = field.type.from_delimited(str_value, pattern_field_delimiter, pattern_context_delimiter)
            else:
                d[field.name] = field.type(str_value)
        return cls(**d)


def _iterate_delimited(path, field_delimiter='\t', pattern_field_delimiter='!', pattern_context_delimiter='_'):
    with open(path, 'rt') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#') or len(line) == 0:
                continue
            yield GeneratedExample.from_delimited(
                line, field_delimiter, pattern_field_delimiter, pattern_context_delimiter)


def number_agreement_data(spacy_tokenize_model, bert_tokenizer, path):
    class_correct = 1
    class_incorrect = 0
    classes = list()
    input_examples = list()
    for example in _iterate_delimited(path):
        words = example.generated_context.split()
        assert(words[example.right_index] == example.form)

        input_example = bert_tokenize_with_spacy_meta(
            spacy_tokenize_model, bert_tokenizer, len(input_examples), words, data_offset=-1)
        input_example.data_ids[example.right_index] = len(input_examples)
        classes.append(class_correct)
        input_examples.append(input_example)

        # switch to the wrong number-agreement
        words[example.right_index] = example.alternate_form

        input_example = bert_tokenize_with_spacy_meta(
            spacy_tokenize_model, bert_tokenizer, len(input_examples), words, data_offset=-1)
        input_example.data_ids[example.right_index] = len(input_examples)
        classes.append(class_incorrect)
        input_examples.append(input_example)

    classes = {'agree': np.array(classes, dtype=np.int32)}
    return RawData(input_examples, classes, validation_proportion_of_train=0.1)
