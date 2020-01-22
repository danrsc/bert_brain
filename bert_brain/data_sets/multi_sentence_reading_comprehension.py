import os
import json
from dataclasses import dataclass

import numpy as np

from .input_features import RawData, KindData, ResponseKind, FieldSpec
from .corpus_base import CorpusBase, CorpusExampleUnifier, path_attribute_field


__all__ = ['MultiSentenceReadingComprehension']


@dataclass(frozen=True)
class MultiSentenceReadingComprehension(CorpusBase):
    path: str = path_attribute_field('multi_sentence_reading_comprehension_path')

    @staticmethod
    def _read_examples(path, example_manager: CorpusExampleUnifier, labels):
        examples = list()

        with open(path, 'rt') as f:
            for line in f:
                fields = json.loads(line.strip('\n'))
                passage = fields['passage']['text']
                passage = passage.split()
                questions = fields['passage']['questions']
                for question in questions:
                    q = question['question'].split()
                    for answer in question['answers']:
                        a = answer['text'].split()
                        if a == ['North', 'to', 'South']:
                            a = ['North', 'to', 'south']
                        elif a == ['7', 'Minutes']:
                            a = ['7', 'minutes']
                        elif a == ['Male', 'Model']:
                            a = ['Male', 'model']
                        elif a == ['Box', 'office', 'Hits']:
                            a = ['Box', 'office', 'hits']
                        elif a == ['The', 'Purple', 'Ball']:
                            a = ['The', 'purple', 'ball']
                        elif a == ['James', 'page', 'jackson']:
                            a = ['James', 'Page', 'Jackson']
                        elif a == ['Around', '3000', 'b.c']:
                            a = ['Around', '3000', 'B.C']
                        elif a == ['23,000', 'b.c']:
                            a = ['23,000', 'B.C']
                        elif a == ['Brean', 'hammond']:
                            a = ['Brean', 'Hammond']
                        elif a == ['Femme', 'Fatale']:
                            a = ['Femme', 'fatale']
                        elif a == ['Bin', 'laden']:
                            a = ['Bin', 'Laden']
                        elif a == ['Urban', 'Farming']:
                            a = ['Urban', 'farming']
                        elif a == ['Natural', 'Resources']:
                            a = ['Natural', 'resources']
                        elif a == ['Curtis,', 'debbie,', 'and', 'steven']:
                            a = ['Curtis,', 'Debbie,', 'and', 'Steven']
                        elif a == ['Curtis,', 'debbie,', 'steven']:
                            a = ['Curtis,', 'Debbie,', 'Steven']
                        elif a == ['14', 'Minutes']:
                            a = ['14', 'minutes']
                        elif a == ['Mandrake', 'Root']:
                            a = ['Mandrake', 'root']
                        elif a == ['Sea', 'Turtles']:
                            a = ['Sea', 'turtles']
                        elif a == ['230,000', 'b.c']:
                            a = ['230,000', 'B.C']
                        elif a == ['Carlos', 'menocal']:
                            a = ['Carlos', 'Menocal']
                        elif a == ['Mediterranean', 'Climates']:
                            a = ['Mediterranean', 'climates']
                        elif a == ['9', 'Minutes']:
                            a = ['9', 'minutes']
                        elif a == ['United', 'states']:
                            a = ['United', 'States']
                        elif a == ['Magical', 'mystery', 'tour']:
                            a = ['Magical', 'Mystery', 'Tour']
                        elif a == ['beatlemania']:
                            a = ['Beatlemania']
                        elif a == ['Index', 'Fossils']:
                            a = ['Index', 'fossils']
                        elif a == ['Guitar', 'wolf']:
                            a = ['Guitar', 'Wolf']
                        elif a == ['Third', 'century', 'b.c']:
                            a = ['Third', 'century', 'B.C']
                        elif a == ['660', 'b.c']:
                            a = ['660', 'B.C']
                        elif a == ['40,000', 'b.c']:
                            a = ['40,000', 'B.C']
                        elif a == ['$', '1.4', 'billion']:
                            a = ['$1.4', 'billion']
                        elif a == ['Jury', 'selection', 'wrapped', 'up', 'Monday,', 'and', 'opening',
                                   'statements', 'were', 'scheduled', 'to', 'begin', 'at', '9', 'a.m.Tuesday']:
                            a = ['Jury', 'selection', 'wrapped', 'up', 'Monday,', 'and', 'opening',
                                 'statements', 'were', 'scheduled', 'to', 'begin', 'at', '9', 'a.m.', 'Tuesday']
                        elif a == ['coffee']:
                            a = ['Coffee']
                        elif a == ['A', 'Mountain']:
                            a = ['A', 'mountain']
                        elif a == ['Black', 'Women']:
                            a = ['Black', 'women']
                        elif a == ['Chris', 'rock']:
                            a = ['Chris', 'Rock']
                        elif a == ['Religious', 'Organizations']:
                            a = ['Religious', 'organizations']
                        elif a == ['510', 'b.c']:
                            a = ['510', 'B.C']

                        if q == ['Where', 'do', 'scientists', 'think', 'the', "earth's", 'magnetic',
                                 'field', 'is', 'generated?']:
                            q = ['Where', 'do', 'scientists', 'think', 'the', "Earth's", 'magnetic',
                                 'field', 'is', 'generated?']
                        elif q == ['Who', 'is', 'the', 'President', 'of', 'Guatemala?']:
                            q = ['Who', 'is', 'the', 'president', 'of', 'Guatemala?']
                        elif q == ['WHat', 'film', 'is', 'John', 'skeptical', 'about?']:
                            q = ['What', 'film', 'is', 'John', 'skeptical', 'about?']
                        elif q == ['Do', 'Magnets', 'stick', 'to', 'all', 'materials?']:
                            q = ['Do', 'magnets', 'stick', 'to', 'all', 'materials?']
                        elif q == ['What', 'was', "Kyle's", "Dad's", 'favorite', 'drink?']:
                            q = ['What', 'was', "Kyle's", "dad's", 'favorite', 'drink?']

                        label = answer['label'] if 'label' in answer else 1
                        data_ids = -1 * np.ones(len(passage) + len(q) + len(a), dtype=np.int64)
                        # doesn't matter which word we attach the label to since we specify below that is_sequence=False
                        data_ids[0] = len(labels)

                        ex = example_manager.add_example(
                            example_key=None,
                            words=passage + q + a,
                            sentence_ids=[0] * len(passage) + [1] * (len(q) + len(a)),
                            data_key='multi_rc',
                            data_ids=data_ids,
                            start=0,
                            stop=len(passage),
                            start_sequence_2=len(passage),
                            stop_sequence_2=len(passage) + len(q) + len(a),
                            allow_duplicates=False)

                        if ex is not None:
                            examples.append(ex)
                            labels.append(label)

            return examples

    @classmethod
    def response_key(cls):
        return 'multi_rc'

    def _load(self, example_manager: CorpusExampleUnifier):
        labels = list()
        train = MultiSentenceReadingComprehension._read_examples(
            os.path.join(self.path, 'train.jsonl'), example_manager, labels)
        validation = MultiSentenceReadingComprehension._read_examples(
            os.path.join(self.path, 'val.jsonl'), example_manager, labels)
        test = MultiSentenceReadingComprehension._read_examples(
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
