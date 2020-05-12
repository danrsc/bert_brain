from dataclasses import dataclass

import numpy as np

from .corpus_base import CorpusBase, CorpusExampleUnifier, path_attribute_field
from .input_features import RawData, KindData, ResponseKind, FieldSpec


__all__ = [
    'BigramShift',
    'CoordinationInversion',
    'ObjectNumber',
    'SemanticOddManOut',
    'SentenceLength',
    'SubjectNumber',
    'TopConstituents',
    'TreeDepth',
    'VerbTense',
    'WordContent']


def _load_probing_task(
        example_manager: CorpusExampleUnifier,
        path: str,
        response_key: str,
        num_classes: int,
        use_meta_train: bool) -> RawData:
    train_examples = list()
    validation_examples = list()
    test_examples = list()
    labels = list()
    with open(path, 'rt') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            fields = line.split('\t')
            if len(fields) != 3:
                raise ValueError('Unexpected number of fields. Expected 3, got {}'.format(len(fields)))
            which, label, sentence = fields
            # not perfect - the sentences are already pre-tokenized, so things like she'd are separated as <she 'd>
            words = sentence.split()
            data_ids = -1 * np.ones(len(words), dtype=np.int64)
            # doesn't matter which word we attach the label to since we specify below that is_sequence=False
            data_ids[0] = len(labels)
            example = example_manager.add_example(
                example_key=sentence,
                words=words,
                sentence_ids=[len(labels)] * len(words),
                data_key=response_key,
                data_ids=data_ids,
                allow_duplicates=False)
            if example is not None:
                if which == 'tr':
                    train_examples.append(example)
                elif which == 'va':
                    validation_examples.append(example)
                elif which == 'te':
                    test_examples.append(example)
                else:
                    raise ValueError('Unexpected data set field: {}'.format(which))
                labels.append(label)

    # convert labels to ids
    unique_labels = set(labels)
    if len(unique_labels) != num_classes:
        raise RuntimeError('number of unique labels ({}) differs from number of classes ({})'.format(
            len(unique_labels), num_classes))
    unique_labels = dict((label, i) for i, label in enumerate(sorted(unique_labels)))
    labels = list(unique_labels[label] for label in labels)

    labels = np.array(labels, dtype=np.float64)
    labels.setflags(write=False)

    meta_train = None
    if use_meta_train:
        from sklearn.model_selection import train_test_split
        idx_train, idx_meta_train = train_test_split(np.arange(len(train_examples)), test_size=0.2)
        meta_train = [train_examples[i] for i in idx_meta_train]
        train_examples = [train_examples[i] for i in idx_train]

    return RawData(
        train_examples,
        response_data={response_key: KindData(ResponseKind.generic, labels)},
        validation_input_examples=validation_examples,
        test_input_examples=test_examples,
        meta_train_input_examples=meta_train,
        is_pre_split=True,
        field_specs={response_key: FieldSpec(is_sequence=False)},
        text_labels={response_key: list(sorted(unique_labels, key=lambda k: unique_labels[k]))})


@dataclass(frozen=True)
class BigramShift(CorpusBase):
    path: str = path_attribute_field('bigram_shift_path')

    @classmethod
    def response_key(cls) -> str:
        return 'bshift'

    @classmethod
    def num_classes(cls) -> int:
        return 2

    def _load(self, example_manager: CorpusExampleUnifier, use_meta_train: bool) -> RawData:
        return _load_probing_task(
            example_manager, self.path, type(self).response_key(), type(self).num_classes(), use_meta_train)


@dataclass(frozen=True)
class CoordinationInversion(CorpusBase):
    path: str = path_attribute_field('coordination_inversion_path')

    @classmethod
    def response_key(cls) -> str:
        return 'coord_inv'

    @classmethod
    def num_classes(cls) -> int:
        return 2

    def _load(self, example_manager: CorpusExampleUnifier, use_meta_train: bool) -> RawData:
        return _load_probing_task(
            example_manager, self.path, type(self).response_key(), type(self).num_classes(), use_meta_train)


@dataclass(frozen=True)
class ObjectNumber(CorpusBase):
    path: str = path_attribute_field('object_number_path')

    @classmethod
    def response_key(cls) -> str:
        return 'obj_num'

    @classmethod
    def num_classes(cls) -> int:
        return 2

    def _load(self, example_manager: CorpusExampleUnifier, use_meta_train: bool) -> RawData:
        return _load_probing_task(
            example_manager, self.path, type(self).response_key(), type(self).num_classes(), use_meta_train)


@dataclass(frozen=True)
class SemanticOddManOut(CorpusBase):
    path: str = path_attribute_field('semantic_odd_man_out_path')

    @classmethod
    def response_key(cls) -> str:
        return 'somo'

    @classmethod
    def num_classes(cls) -> int:
        return 2

    def _load(self, example_manager: CorpusExampleUnifier, use_meta_train) -> RawData:
        return _load_probing_task(
            example_manager, self.path, type(self).response_key(), type(self).num_classes(), use_meta_train)


@dataclass(frozen=True)
class SentenceLength(CorpusBase):
    path: str = path_attribute_field('sentence_length_path')

    @classmethod
    def response_key(cls) -> str:
        return 'sent_len'

    @classmethod
    def num_classes(cls) -> int:
        return 6

    def _load(self, example_manager: CorpusExampleUnifier, use_meta_train: bool) -> RawData:
        return _load_probing_task(
            example_manager, self.path, type(self).response_key(), type(self).num_classes(), use_meta_train)


@dataclass(frozen=True)
class SubjectNumber(CorpusBase):
    path: str = path_attribute_field('subject_number_path')

    @classmethod
    def response_key(cls) -> str:
        return 'subj_num'

    @classmethod
    def num_classes(cls) -> int:
        return 2

    def _load(self, example_manager: CorpusExampleUnifier, use_meta_train: bool) -> RawData:
        return _load_probing_task(
            example_manager, self.path, type(self).response_key(), type(self).num_classes(), use_meta_train)


@dataclass(frozen=True)
class TopConstituents(CorpusBase):
    path: str = path_attribute_field('top_constituents_path')

    @classmethod
    def response_key(cls) -> str:
        return 'top_const'

    @classmethod
    def num_classes(cls) -> int:
        return 20

    def _load(self, example_manager: CorpusExampleUnifier, use_meta_train: bool) -> RawData:
        return _load_probing_task(
            example_manager, self.path, type(self).response_key(), type(self).num_classes(), use_meta_train)


@dataclass(frozen=True)
class TreeDepth(CorpusBase):
    path: str = path_attribute_field('tree_depth_path')

    @classmethod
    def response_key(cls) -> str:
        return 'tree_depth'

    @classmethod
    def num_classes(cls) -> int:
        return 7

    def _load(self, example_manager: CorpusExampleUnifier, use_meta_train: bool) -> RawData:
        return _load_probing_task(
            example_manager, self.path, type(self).response_key(), type(self).num_classes(), use_meta_train)


@dataclass(frozen=True)
class VerbTense(CorpusBase):
    path: str = path_attribute_field('verb_tense_path')

    @classmethod
    def response_key(cls) -> str:
        return 'tense'

    @classmethod
    def num_classes(cls) -> int:
        return 2

    def _load(self, example_manager: CorpusExampleUnifier, use_meta_train: bool) -> RawData:
        return _load_probing_task(
            example_manager, self.path, type(self).response_key(), type(self).num_classes(), use_meta_train)


@dataclass(frozen=True)
class WordContent(CorpusBase):
    path: str = path_attribute_field('word_content_path')

    @classmethod
    def response_key(cls) -> str:
        return 'wc'

    @classmethod
    def num_classes(cls) -> int:
        return 1000

    def _load(self, example_manager: CorpusExampleUnifier, use_meta_train: bool) -> RawData:
        return _load_probing_task(
            example_manager, self.path, type(self).response_key(), type(self).num_classes(), use_meta_train)
