import os
import json
from dataclasses import dataclass

import numpy as np

from .input_features import RawData, KindData, ResponseKind, FieldSpec
from .corpus_base import CorpusBase, CorpusExampleUnifier, path_attribute_field


__all__ = ['ChoiceOfPlausibleAlternatives']


@dataclass(frozen=True)
class ChoiceOfPlausibleAlternatives(CorpusBase):
    path: str = path_attribute_field('choice_of_plausible_alternatives_path')

    @staticmethod
    def _read_examples(path, example_manager: CorpusExampleUnifier, labels):
        examples = list()
        with open(path, 'rt') as f:
            for line in f:
                fields = json.loads(line.strip('\n'))
                premise = fields['premise'].split()
                multipart_id = len(example_manager)
                choices = list()
                while True:
                    choice_name = 'choice{}'.format(len(choices) + 1)
                    if choice_name not in fields:
                        break
                    choices.append(fields[choice_name].split())
                question_expansions = {
                    'cause': 'What was the cause of this?',
                    'effect': 'What happened as a result?'}
                if fields['question'] not in question_expansions:
                    raise ValueError('Uknown question type: {}'.format(fields['question']))
                question = question_expansions[fields['question']].split()
                label = fields['label'] if 'label' in fields else 1
                for index_choice, choice in enumerate(choices):
                    data_ids = -1 * np.ones(len(premise) + len(question) + len(choice), dtype=np.int64)
                    # doesn't matter which word we attach the label to since we specify below that is_sequence=False
                    data_ids[0] = len(labels)
                    choice_label = 1 if label == index_choice else 0
                    examples.append(example_manager.add_example(
                        example_key=None,
                        words=premise + question + choice,
                        sentence_ids=[0] * len(premise) + [1] * len(question) + [2] * len(choice),
                        data_key='copa',
                        data_ids=data_ids,
                        start=0,
                        stop=len(premise),
                        start_sequence_2=len(premise),
                        stop_sequence_2=len(premise) + len(question),
                        start_sequence_3=len(premise) + len(question),
                        stop_sequence_3=len(premise) + len(question) + len(choice),
                        multipart_id=multipart_id))
                    labels.append(choice_label)
        return examples

    @classmethod
    def response_key(cls) -> str:
        return 'copa'

    @classmethod
    def num_classes(cls) -> int:
        return 2

    def _load(self, example_manager: CorpusExampleUnifier, use_meta_train: bool):
        labels = list()
        train = ChoiceOfPlausibleAlternatives._read_examples(
            os.path.join(self.path, 'train.jsonl'), example_manager, labels)
        meta_train = None
        if use_meta_train:
            from sklearn.model_selection import train_test_split
            idx_train, idx_meta_train = train_test_split(np.arange(len(train)), test_size=0.2)
            meta_train = [train[i] for i in idx_meta_train]
            train = [train[i] for i in idx_train]
        validation = ChoiceOfPlausibleAlternatives._read_examples(
            os.path.join(self.path, 'val.jsonl'), example_manager, labels)
        test = ChoiceOfPlausibleAlternatives._read_examples(
            os.path.join(self.path, 'test.jsonl'), example_manager, labels)
        labels = np.array(labels, dtype=np.float64)
        labels.setflags(write=False)
        return RawData(
            input_examples=train,
            validation_input_examples=validation,
            test_input_examples=test,
            meta_train_input_examples=meta_train,
            response_data={type(self).response_key(): KindData(ResponseKind.generic, labels)},
            is_pre_split=True,
            field_specs={type(self).response_key(): FieldSpec(is_sequence=False)})
