import os
import json
import string
from dataclasses import dataclass

import numpy as np

from ..common import NamedSpanEncoder
from .input_features import RawData, KindData, ResponseKind, FieldSpec
from .corpus_base import CorpusBase, CorpusExampleUnifier, path_attribute_field


__all__ = ['WinogradSchemaChallenge']


_punctuation_set = set(string.punctuation)


def _strip_punctuation_apostrophe(s):
    start = 0
    while s[start] in _punctuation_set:
        start += 1
    end = len(s)
    while s[end - 1] in _punctuation_set:
        end -= 1
    s = s[start:end]
    if s.endswith("'s"):
        return s[:-len("'s")]
    return s


@dataclass(frozen=True)
class WinogradSchemaChallenge(CorpusBase):
    path: str = path_attribute_field('winograd_schema_challenge_path')

    @staticmethod
    def _read_examples(path, example_manager: CorpusExampleUnifier, labels, named_span_encoder):
        examples = list()
        with open(path, 'rt') as f:
            for line in f:
                try:
                    fields = json.loads(line.strip('\n'))
                    text = fields['text'].split()
                    span_ids = [0] * len(text)
                    target = fields['target']
                    span_id = 1
                    while True:
                        span_index_field = 'span{}_index'.format(span_id)
                        span_text_field = 'span{}_text'.format(span_id)
                        if span_index_field not in target:
                            break
                        span_index = target[span_index_field]
                        span_text = target[span_text_field].split()
                        encoded_span = named_span_encoder.encode('span_{}'.format(span_id))
                        for i in range(len(span_text)):
                            if span_text[i].lower() != text[i + span_index].lower() \
                                    and (_strip_punctuation_apostrophe(span_text[i].lower())
                                         != _strip_punctuation_apostrophe(text[i + span_index].lower())):
                                if (text == ['When', 'they', 'had', 'eventually', 'calmed', 'down', 'a', 'bit', ',',
                                             'and', 'had', 'gotten', 'home,', 'Mr.', 'Farley', 'put', 'the', 'magic',
                                             'pebble', 'in', 'an', 'iron', 'safe', '.', 'Some', 'day', 'they', 'might',
                                             'want', 'to', 'use', 'it', ',', 'but', 'really', 'for', 'now,', 'what',
                                             'more', 'could', 'they', 'wish', 'for?']
                                        and span_index == 30
                                        and span_text == ['it']
                                        and text[span_index:span_index + len(span_text)] == ['use']):
                                    span_index = 31
                                    assert(text[span_index:span_index + len(span_text)] == span_text)
                                else:
                                    raise ValueError('Mismatched span')
                            span_ids[i + span_index] += encoded_span
                        span_id += 1
                    label = fields['label'] if 'label' in fields else 1
                    data_ids = -1 * np.ones(len(text), dtype=np.int64)
                    # doesn't matter which word we attach the label to since we specify below that is_sequence=False
                    data_ids[0] = len(labels)

                    ex = example_manager.add_example(
                        example_key=tuple(text + [str(n) for n in span_ids]),
                        words=text,
                        sentence_ids=[0] * len(text),
                        data_key='wsc',
                        data_ids=data_ids,
                        span_ids=span_ids,
                        start=0,
                        stop=len(text),
                        allow_duplicates=False)
                    if ex is not None:
                        examples.append(ex)
                        labels.append(label)
                except ValueError as e:
                    if (str(e) == 'Mismatched span'
                            and text == ['Kotkin', 'describes', 'how', 'Lev', 'Kamenev,', "Stalin's", 'old', 'Pravda',
                                         'co-editor,', 'and', 'Grigory', 'Zinoviev,', 'who', 'with', 'Stalin', 'and',
                                         'Kamenev', 'had', 'formed', 'a', 'ruling', 'troika', 'during', "Lenin's",
                                         'final', 'illness,', 'were', 'dragged', 'out', 'of', 'their', 'prison',
                                         'cells', 'in', '1936', 'for', 'a', 'meeting', 'with', 'Stalin;', 'he',
                                         'urged', 'them', 'to', 'confess,', 'for', 'old', "times'", 'sake.']):
                        continue
                    else:
                        raise
        return examples

    @classmethod
    def response_key(cls):
        return 'wsc'

    @classmethod
    def num_classes(cls):
        return 2

    def _load(self, example_manager: CorpusExampleUnifier):
        labels = list()
        named_span_encoder = NamedSpanEncoder()
        train = WinogradSchemaChallenge._read_examples(
            os.path.join(self.path, 'train.jsonl'), example_manager, labels, named_span_encoder)
        validation = WinogradSchemaChallenge._read_examples(
            os.path.join(self.path, 'val.jsonl'), example_manager, labels, named_span_encoder)
        test = WinogradSchemaChallenge._read_examples(
            os.path.join(self.path, 'test.jsonl'), example_manager, labels, named_span_encoder)
        labels = np.array(labels, dtype=np.float64)
        labels.setflags(write=False)
        return RawData(
            input_examples=train,
            validation_input_examples=validation,
            test_input_examples=test,
            response_data={type(self).response_key(): KindData(ResponseKind.generic, labels)},
            is_pre_split=True,
            field_specs={type(self).response_key(): FieldSpec(is_sequence=False)})
