import os
import json

import numpy as np

from ..common import split_with_indices
from .input_features import RawData, KindData, ResponseKind
from .corpus_base import CorpusBase, CorpusExampleUnifier


__all__ = ['WordInContext']


class WordInContext(CorpusBase):

    @classmethod
    def _path_attributes(cls):
        return dict(path='word_in_context_path')

    def __init__(self, path=None):
        self.path = path

    @staticmethod
    def sentence_and_keyword_index(sentence, keyword, character_index):
        keyword_index = None
        words = list()
        for w_index, (c_index, word) in enumerate(split_with_indices(sentence)):
            if c_index + len(word) > character_index >= c_index:
                keyword_index = w_index
            words.append(word)
        if keyword_index is None:
            raise ValueError('Unable to match keyword index')
        return words, keyword_index

    @staticmethod
    def _read_examples(path, example_manager: CorpusExampleUnifier, labels):
        examples = list()
        with open(path, 'rt') as f:
            for line in f:
                fields = json.loads(line.strip('\n'))
                words_1, keyword_1 = WordInContext.sentence_and_keyword_index(
                    fields['sentence1'], fields['word'], fields['start1'])
                words_2, keyword_2 = WordInContext.sentence_and_keyword_index(
                    fields['sentence2'], fields['word'], fields['start2'])
                label = fields['label'] if 'label' in fields else 1
                data_ids = -1 * np.ones(len(words_1) + len(words_2), dtype=np.int64)
                data_ids[keyword_1] = len(labels)
                data_ids[keyword_2] = len(labels)
                examples.append(example_manager.add_example(
                    example_key=None,
                    words=words_1 + words_2,
                    sentence_ids=[0] * len(words_1) + [1] * len(words_2),
                    data_key='wic',
                    data_ids=data_ids,
                    start=0,
                    stop=len(words_1),
                    start_sequence_2=len(words_1),
                    stop_sequence_2=len(words_1) + len(words_2)))
                labels.append(label)
        return examples

    @classmethod
    def response_key(cls):
        return 'wic'

    def _load(self, run_info, example_manager: CorpusExampleUnifier):
        labels = list()
        train = WordInContext._read_examples(
            os.path.join(self.path, 'train.jsonl'), example_manager, labels)
        validation = WordInContext._read_examples(
            os.path.join(self.path, 'val.jsonl'), example_manager, labels)
        test = WordInContext._read_examples(
            os.path.join(self.path, 'test.jsonl'), example_manager, labels)
        labels = np.array(labels, dtype=np.float64)
        labels.setflags(write=False)
        return RawData(
            input_examples=train,
            validation_input_examples=validation,
            test_input_examples=test,
            response_data={type(self).response_key(): KindData(ResponseKind.generic, labels)},
            is_pre_split=True)
