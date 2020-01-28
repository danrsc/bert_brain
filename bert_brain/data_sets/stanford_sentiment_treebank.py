import os
from dataclasses import dataclass

import numpy as np

from .corpus_base import CorpusBase, CorpusExampleUnifier, path_attribute_field
from .input_features import RawData, KindData, ResponseKind, FieldSpec


__all__ = ['StanfordSentimentTreebank']


@dataclass(frozen=True)
class StanfordSentimentTreebank(CorpusBase):
    path: str = path_attribute_field('stanford_sentiment_treebank_path')

    @staticmethod
    def _read_labels(label_list, example_manager: CorpusExampleUnifier, path: str):
        examples = list()
        with open(path, 'rt') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                fields = line.split('\t')
                if len(fields) != 2:
                    raise ValueError('Unexpected number of fields. Expected 2, got {}'.format(len(fields)))
                sentence, label = fields
                if sentence == 'sentence' and label == 'label':  # header
                    continue
                # not sure about this - from what I can tell, it seems like BERT just uses the treebank parse
                # as is without combining things like <do n't> back together
                words = sentence.split()
                data_ids = -1 * np.ones(len(words), dtype=np.int64)
                # doesn't matter which word we attach the label to since we specify below that is_sequence=False
                data_ids[0] = len(label_list)
                examples.append(example_manager.add_example(
                    example_key=sentence,
                    words=words,
                    sentence_ids=[len(label_list)] * len(words),
                    data_key='sentiment',
                    data_ids=data_ids))
                label_list.append(label)
        return examples

    @classmethod
    def response_key(cls) -> str:
        return 'sentiment'

    @classmethod
    def num_classes(cls) -> int:
        return 2

    def _load(self, example_manager: CorpusExampleUnifier):
        label_list = list()
        train_examples = StanfordSentimentTreebank._read_labels(
            label_list, example_manager, os.path.join(self.path, 'train.tsv'))
        validation_examples = StanfordSentimentTreebank._read_labels(
            label_list, example_manager, os.path.join(self.path, 'dev.tsv'))

        labels = np.array(label_list, dtype=np.float64)
        labels.setflags(write=False)

        return RawData(
            train_examples,
            response_data={type(self).response_key(): KindData(ResponseKind.generic, labels)},
            validation_input_examples=validation_examples,
            is_pre_split=True,
            field_specs={type(self).response_key(): FieldSpec(is_sequence=False)})
