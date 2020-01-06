import os
import json

import numpy as np

from .input_features import RawData, KindData, ResponseKind, FieldSpec
from .corpus_base import CorpusBase, CorpusExampleUnifier
from .spacy_token_meta import ChineseCharDetected


__all__ = ['BooleanQuestions']


class BooleanQuestions(CorpusBase):

    @classmethod
    def _path_attributes(cls):
        return dict(path='boolq_path')

    def __init__(self, path=None):
        self.path = path

    @staticmethod
    def _read_examples(path, example_manager: CorpusExampleUnifier, labels):
        examples = list()
        with open(path, 'rt') as f:
            for line in f:
                fields = json.loads(line.strip('\n'))
                passage = fields['passage'].split()
                question = fields['question'].split()
                label = fields['label'] if 'label' in fields else True
                question_ = list()
                for w in question:
                    if w == 'fitness(as':
                        question_.extend(['fitness', '(as'])
                    else:
                        question_.append(w)
                question = question_
                data_ids = -1 * np.ones(len(passage) + len(question), dtype=np.int64)
                # doesn't matter which word we attach the label to since we specify below that is_sequence=False
                data_ids[0] = len(labels)
                try:
                    ex = example_manager.add_example(
                        example_key=None,
                        words=passage + question,
                        sentence_ids=[0] * len(passage) + [1] * len(question),
                        data_key='boolq',
                        data_ids=data_ids,
                        start=0,
                        stop=len(passage),
                        start_sequence_2=len(passage),
                        stop_sequence_2=len(passage) + len(question),
                        allow_duplicates=False)

                    if ex is not None:
                        examples.append(ex)
                        labels.append(label)

                except ChineseCharDetected:
                    # 64 of 9363 training examples eliminated (0.7%)
                    pass

        return examples

    @classmethod
    def response_key(cls):
        return 'boolq'

    def _load(self, run_info, example_manager: CorpusExampleUnifier):
        labels = list()
        train = BooleanQuestions._read_examples(
            os.path.join(self.path, 'train.jsonl'), example_manager, labels)
        validation = BooleanQuestions._read_examples(
            os.path.join(self.path, 'val.jsonl'), example_manager, labels)
        test = BooleanQuestions._read_examples(
            os.path.join(self.path, 'test.jsonl'), example_manager, labels)
        labels = np.array(labels, dtype=np.float64)
        labels.setflags(write=False)
        return RawData(
            input_examples=train,
            validation_input_examples=validation,
            test_input_examples=test,
            response_data={type(self).response_key(): KindData(ResponseKind.generic, labels)},
            is_pre_split=True,
            field_specs={type(self).response_key(): FieldSpec(is_sequence=False)})
