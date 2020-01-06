import os
from collections import OrderedDict
import json
import re
import string

import numpy as np

from .input_features import RawData, KindData, ResponseKind, FieldSpec
from .corpus_base import CorpusBase, CorpusExampleUnifier
from .spacy_token_meta import ChineseCharDetected


__all__ = ['ReadingComprehensionWithCommonSenseReasoning']


_punctuation_set = set(string.punctuation)


class ReadingComprehensionWithCommonSenseReasoning(CorpusBase):

    @classmethod
    def _path_attributes(cls):
        return dict(path='reading_comprehension_with_common_sense_reasoning_path')

    def __init__(self, path=None):
        self.path = path

    @staticmethod
    def _normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace.
        From official ReCoRD eval script """

        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            return "".join(ch for ch in text if ch not in _punctuation_set)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @staticmethod
    def _token_f1_em(prediction, target):
        prediction = ReadingComprehensionWithCommonSenseReasoning._normalize_answer(prediction).split()
        target = ReadingComprehensionWithCommonSenseReasoning._normalize_answer(target).split()
        common = set(prediction).intersection(target)
        if len(common) == 0:
            return 0, False
        precision = len(common) / len(prediction)
        recall = len(common) / len(target)
        return (2 * precision * recall) / (precision + recall), prediction == target

    @staticmethod
    def _read_examples(path, example_manager: CorpusExampleUnifier, labels, f1_list):
        examples = list()
        with open(path, 'rt') as f:
            for line in f:
                fields = json.loads(line.strip('\n'))
                passage_ = fields['passage']['text']
                entities = OrderedDict()
                unique_entities = set()
                for entity_indices in fields['passage']['entities']:
                    start, end = entity_indices['start'], entity_indices['end']
                    entities[(start, end)] = passage_[start:(end + 1)]
                    unique_entities.add(passage_[start:(end + 1)])
                unique_entities_ = set()
                for entity in unique_entities:
                    if entity.lower() not in unique_entities:
                        unique_entities_.add(entity)
                unique_entities = unique_entities_
                for question_answer in fields['qas']:
                    multipart_id = len(example_manager)
                    question_template = question_answer['query']
                    answer_entities = set()
                    if 'answers' in question_answer:
                        for answer_indices in question_answer['answers']:
                            answer_entities.add(entities[(answer_indices['start'], answer_indices['end'])].lower())
                    for entity in unique_entities:
                        max_f1 = 0
                        label = 0
                        for answer_entity in answer_entities:
                            f1, is_exact = ReadingComprehensionWithCommonSenseReasoning._token_f1_em(
                                entity, answer_entity)
                            max_f1 = max(f1, max_f1)
                            if is_exact:
                                label = 1
                        question = question_template.replace('@placeholder', entity)

                        question = question.split()
                        passage = passage_.split()
                        data_ids = -1 * np.ones(len(passage) + len(question), dtype=np.int64)
                        # doesn't matter which word we attach the label to since we specify below that is_sequence=False
                        data_ids[0] = len(labels)

                        try:
                            ex = example_manager.add_example(
                                example_key=None,
                                words=passage + question,
                                sentence_ids=[0] * len(passage) + [1] * len(question),
                                data_key='record',
                                data_ids=data_ids,
                                start=0,
                                stop=len(passage),
                                start_sequence_2=len(passage),
                                stop_sequence_2=len(passage) + len(question),
                                multipart_id=multipart_id,
                                allow_duplicates=False,
                                auto_high_frequency_on_collision=True)

                            if ex is not None:
                                examples.append(ex)
                                labels.append(label)
                                f1_list.append(max_f1)
                        except ChineseCharDetected:
                            pass
        return examples

    @classmethod
    def response_key(cls):
        return 'record'

    def _load(self, run_info, example_manager: CorpusExampleUnifier):
        labels = list()
        f1 = list()
        train = ReadingComprehensionWithCommonSenseReasoning._read_examples(
            os.path.join(self.path, 'train.jsonl'), example_manager, labels, f1)
        validation = ReadingComprehensionWithCommonSenseReasoning._read_examples(
            os.path.join(self.path, 'val.jsonl'), example_manager, labels, f1)
        test = ReadingComprehensionWithCommonSenseReasoning._read_examples(
            os.path.join(self.path, 'test.jsonl'), example_manager, labels, f1)
        labels = np.array(labels, dtype=np.float64)
        labels.setflags(write=False)
        f1 = np.array(f1, dtype=np.float64)
        f1.setflags(write=False)
        return RawData(
            input_examples=train,
            validation_input_examples=validation,
            test_input_examples=test,
            response_data={type(self).response_key(): KindData(ResponseKind.generic, labels)},
            is_pre_split=True,
            field_specs={type(self).response_key(): FieldSpec(is_sequence=False)},
            metadata=dict(f1=f1))
