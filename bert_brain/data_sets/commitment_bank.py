import os
import json
from dataclasses import dataclass

import numpy as np

from .input_features import RawData, KindData, ResponseKind, FieldSpec
from .corpus_base import CorpusBase, CorpusExampleUnifier, path_attribute_field


__all__ = ['CommitmentBank']


@dataclass(frozen=True)
class CommitmentBank(CorpusBase):
    path: str = path_attribute_field('commitment_bank_path')

    @staticmethod
    def _read_examples(path, example_manager: CorpusExampleUnifier, labels, classes):
        examples = list()
        with open(path, 'rt') as f:
            for line in f:
                fields = json.loads(line.strip('\n'))
                premise = fields['premise'].split()
                hypothesis = fields['hypothesis'].split()
                label = fields['label'] if 'label' in fields else 'unknown'
                if label not in classes:
                    raise ValueError('Unknown label: {}'.format(label))
                label = classes[label]
                data_ids = -1 * np.ones(len(premise) + len(hypothesis), dtype=np.int64)
                # doesn't matter which word we attach the label to since we specify below that is_sequence=False
                data_ids[0] = len(labels)
                examples.append(example_manager.add_example(
                    example_key=None,
                    words=premise + hypothesis,
                    sentence_ids=[0] * len(premise) + [1] * len(hypothesis),
                    data_key='cb',
                    data_ids=data_ids,
                    start=0,
                    stop=len(premise),
                    start_sequence_2=len(premise),
                    stop_sequence_2=len(premise) + len(hypothesis)))
                labels.append(label)
        return examples

    @classmethod
    def response_key(cls) -> str:
        return 'cb'

    @classmethod
    def num_classes(cls) -> int:
        return 4

    def _load(self, example_manager: CorpusExampleUnifier, use_meta_train: bool):
        classes = {
            'unknown': 0,
            'entailment': 1,
            'contradiction': 2,
            'neutral': 3
        }
        labels = list()
        train = CommitmentBank._read_examples(
            os.path.join(self.path, 'train.jsonl'), example_manager, labels, classes)
        meta_train = None
        if use_meta_train > 0:
            from sklearn.model_selection import train_test_split
            idx_train, idx_meta_train = train_test_split(np.arange(len(train)), test_size=0.2)
            meta_train = [train[i] for i in idx_meta_train]
            train = [train[i] for i in idx_train]
        validation = CommitmentBank._read_examples(
            os.path.join(self.path, 'val.jsonl'), example_manager, labels, classes)
        test = CommitmentBank._read_examples(
            os.path.join(self.path, 'test.jsonl'), example_manager, labels, classes)
        labels = np.array(labels, dtype=np.float64)
        labels.setflags(write=False)
        return RawData(
            input_examples=train,
            validation_input_examples=validation,
            test_input_examples=test,
            meta_train_input_examples=meta_train,
            response_data={type(self).response_key(): KindData(ResponseKind.generic, labels)},
            is_pre_split=True,
            field_specs={type(self).response_key(): FieldSpec(is_sequence=False)},
            text_labels={type(self).response_key(): list(sorted(classes, key=lambda k: classes[k]))})
