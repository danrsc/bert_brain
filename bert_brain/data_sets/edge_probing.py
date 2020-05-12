import os
from dataclasses import dataclass
from collections import OrderedDict
from typing import Callable, Optional, ClassVar
import json

import numpy as np
from tqdm.auto import tqdm

from .corpus_base import CorpusBase, CorpusExampleUnifier, path_attribute_field
from .input_features import RawData, KindData, ResponseKind, FieldSpec
from .spacy_token_meta import ChineseCharDetected
from ..common import NamedSpanEncoder


__all__ = [
    'PartOfSpeechConll2012',
    'SimplifiedPartOfSpeechConll2012',
    'ConstituentsConll2012',
    'SemanticRoleLabelConll2012',
    'NamedEntityRecognitionConll2012',
    'CoreferenceResolutionConll2012',
    'DependenciesEnglishWeb',
    'DefinitePronounResolution',
    'SemEval',
    'SemanticProtoRoles1',
    'SemanticProtoRoles2']


def _read_records(path: str, named_span_encoder: NamedSpanEncoder):
    span1_value = named_span_encoder.encode('span1')
    span2_value = None
    with open(path, 'rt') as f:
        for line in tqdm(f, desc='records'):
            record = json.loads(line)
            words = record['text'].split()
            for target in record['targets']:
                span_ids = [0] * len(words)
                span1 = target['span1']
                for i in range(span1[0], span1[1]):
                    span_ids[i] += span1_value
                labels = target['label']
                if 'span2' in target:
                    if span2_value is None:
                        span2_value = named_span_encoder.encode('span2')
                    span2 = target['span2']
                    for i in range(span2[0], span2[1]):
                        span_ids[i] += span2_value
                yield words, span_ids, labels


def _store_records_and_find_labels(label_choices_to_expand: set, path: str, named_span_encoder: NamedSpanEncoder):
    all_words = list()
    all_spans = list()
    all_labels = list()
    for words, span_ids, labels in _read_records(path, named_span_encoder):
        all_words.append(words)
        all_spans.append(span_ids)
        all_labels.append(labels)
        for label in labels:
            label_choices_to_expand.add(label)
    return all_words, all_spans, all_labels


def _load_multi_binary_label_probing_task(
        example_manager: CorpusExampleUnifier,
        path: str,
        response_key: str,
        validation_path: Optional[str] = None,
        test_path: Optional[str] = None,
        use_meta_train: bool = False) -> RawData:
    label_choices = set()
    named_span_encoder = NamedSpanEncoder()

    if (validation_path is None) != (test_path is None):
        raise ValueError('If either validaition_path or test_path is given, both must be given')

    if validation_path is None:
        train_path = os.path.join(path, 'train.json')
        validation_path = os.path.join(path, 'development.json')
        test_path = os.path.join(path, 'test.json')
    else:
        train_path = path

    train_words, train_spans, train_true = _store_records_and_find_labels(
        label_choices, train_path, named_span_encoder)
    validation_words, validation_spans, validation_true = _store_records_and_find_labels(
        label_choices, validation_path, named_span_encoder)
    test_words, test_spans, test_true = _store_records_and_find_labels(
        label_choices, test_path, named_span_encoder)

    label_choices = list(sorted(label_choices))
    train_examples = list()
    validation_examples = list()
    test_examples = list()
    output_labels = OrderedDict((k, list()) for k in label_choices)

    for all_words, all_spans, all_true, example_list in [
            (train_words, train_spans, train_true, train_examples),
            (validation_words, validation_spans, validation_true, validation_examples),
            (test_words, test_spans, test_true, test_examples)]:
        for words, span_ids, true_labels in zip(all_words, all_spans, all_true):
            data_ids = -1 * np.ones(len(words), dtype=np.int64)
            # doesn't matter which word we attach the label to since we specify below that is_sequence=False
            data_ids[0] = len(output_labels[label_choices[0]])
            try:
                ex = example_manager.add_example(
                    example_key=data_ids[0],
                    words=words,
                    sentence_ids=[data_ids[0]] * len(words),
                    data_key=['{}.{}'.format(response_key, choice) for choice in label_choices],
                    data_ids=data_ids,
                    span_ids=span_ids)
                for choice in label_choices:
                    output_labels[choice].append(1 if choice in true_labels else 0)
                example_list.append(ex)
            except ChineseCharDetected:
                pass

    output_labels = OrderedDict(
        ('{}.{}'.format(response_key, k),
         KindData(ResponseKind.generic, np.array(output_labels[k], dtype=np.float64))) for k in output_labels)
    for k in output_labels:
        output_labels[k].data.setflags(write=False)

    meta_train = None
    if use_meta_train:
        from sklearn.model_selection import train_test_split
        idx_train, idx_meta_train = train_test_split(np.arange(len(train_examples)), test_size=0.2)
        meta_train = [train_examples[i] for i in idx_meta_train]
        train_examples = [train_examples[i] for i in idx_train]

    return RawData(
        train_examples,
        response_data=output_labels,
        validation_input_examples=validation_examples,
        test_input_examples=test_examples,
        meta_train_input_examples=meta_train,
        is_pre_split=True,
        field_specs=dict((k, FieldSpec(is_sequence=False)) for k in output_labels))


def _load_probing_task(
        example_manager: CorpusExampleUnifier,
        path: str,
        response_key: str,
        num_classes: int,
        label_transform: Optional[Callable[[str], str]] = None,
        validation_path: Optional[str] = None,
        test_path: Optional[str] = None,
        use_meta_train: bool = False) -> RawData:

    train_examples = list()
    validation_examples = list()
    test_examples = list()
    labels = list()
    named_span_encoder = NamedSpanEncoder()

    if (validation_path is None) != (test_path is None):
        raise ValueError('If either validaition_path or test_path is given, both must be given')

    if validation_path is None:
        train_path = os.path.join(path, 'train.json')
        validation_path = os.path.join(path, 'development.json')
        test_path = os.path.join(path, 'test.json')
    else:
        train_path = path

    for path, example_list in [
            (train_path, train_examples), (validation_path, validation_examples), (test_path, test_examples)]:
        for words, span_ids, label in _read_records(path, named_span_encoder):
            data_ids = -1 * np.ones(len(words), dtype=np.int64)
            # doesn't matter which word we attach the label to since we specify below that is_sequence=False
            data_ids[0] = len(labels)
            try:
                example_list.append(example_manager.add_example(
                    example_key=len(labels),
                    words=words,
                    sentence_ids=[len(labels)] * len(words),
                    data_key=response_key,
                    data_ids=data_ids,
                    span_ids=span_ids))
                if label_transform is not None:
                    label = label_transform(label)
                labels.append(label)
            except ChineseCharDetected:
                pass

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
class PartOfSpeechConll2012(CorpusBase):
    path: str = path_attribute_field('part_of_speech_conll_2012_path')

    @classmethod
    def response_key(cls) -> str:
        return 'pos_conll'

    @classmethod
    def num_classes(cls) -> int:
        return 48

    @classmethod
    def num_spans(cls) -> int:
        return 1

    def _load(self, example_manager: CorpusExampleUnifier, use_meta_train: bool) -> RawData:
        return _load_probing_task(
            example_manager, self.path, type(self).response_key(), type(self).num_classes(),
            use_meta_train=use_meta_train)


@dataclass(frozen=True)
class SimplifiedPartOfSpeechConll2012(CorpusBase):
    path: str = path_attribute_field('part_of_speech_conll_2012_path')

    # https://spacy.io/api/annotation
    _pos_map: ClassVar[dict] = {
        '$': 'SYM',         #                                                       symbol, currency
        "``": 'PUNCT',      #   PunctType=quot PunctSide=ini                        opening quotation mark
        "''": 'PUNCT',      #   PunctType=quot PunctSide=fin                        closing quotation mark
        ',': 'PUNCT',       #   PunctType=comm	                                    punctuation mark, comma
        '-LRB-': 'PUNCT',   #   PunctType=brck PunctSide=ini	                    left round bracket
        '-RRB-': 'PUNCT',   # 	PunctType=brck PunctSide=fin	                    right round bracket
        '.': 'PUNCT',       # 	PunctType=peri	                                    punctuation mark, sentence closer
        ':': 'PUNCT',       # 		                                                punctuation mark, colon or ellipsis
        'ADD': 'X',         # 		                                                email
        'AFX': 'ADJ',       # 	Hyph=yes	                                        affix
        'CC': 'CCONJ',      # 	ConjType=comp	                                    conjunction, coordinating
        'CD': 'NUM',        # 	NumType=card	                                    cardinal number
        'DT': 'DET',        # 		                                                determiner
        'EX': 'PRON',       # 	AdvType=ex	                                        existential there
        'FW': 'X',          # 	Foreign=yes	                                        foreign word
        'GW': 'X',          # 		                                                additional word in multi-word expression
        'HYPH': 'PUNCT',    # 	PunctType=dash	                                    punctuation mark, hyphen
        'IN': 'ADP',        # 		                                                conjunction, subordinating or preposition
        'JJ': 'ADJ',        # 	Degree=pos	                                        adjective
        'JJR': 'ADJ',       # 	Degree=comp	                                        adjective, comparative
        'JJS': 'ADJ',       # 	Degree=sup	                                        adjective, superlative
        'LS': 'X',          # 	NumType=ord	                                        list item marker
        'MD': 'VERB',       # 	VerbType=mod	                                    verb, modal auxiliary
        'NFP': 'PUNCT',     # 		                                                superfluous punctuation
        'NIL': 'X',         # 		                                                missing tag
        'NN': 'NOUN',       # 	Number=sing	                                        noun, singular or mass
        'NNP': 'PROPN',     # 	NounType=prop Number=sing	                        noun, proper singular
        'NNPS': 'PROPN',    # 	NounType=prop Number=plur	                        noun, proper plural
        'NNS': 'NOUN',      # 	Number=plur	                                        noun, plural
        'PDT': 'DET',       # 		                                                predeterminer
        'POS': 'PART',      # 	Poss=yes	                                        possessive ending
        'PRP': 'PRON',      # 	PronType=prs	                                    pronoun, personal
        'PRP$': 'DET',      # 	PronType=prs Poss=yes	                            pronoun, possessive
        'RB': 'ADV',        # 	Degree=pos	                                        adverb
        'RBR': 'ADV',       #   Degree=comp	                                        adverb, comparative
        'RBS': 'ADV',       #   Degree=sup	                                        adverb, superlative
        'RP': 'ADP',        #                                                       adverb, particle
        'SP': 'SPACE',      # 		                                                space
        'SYM': 'SYM',       # 		                                                symbol
        'TO': 'PART',       # 	PartType=inf VerbForm=inf	                        infinitival “to”
        'UH': 'INTJ',       # 		                                                interjection
        'VB': 'VERB',       # 	VerbForm=inf	                                    verb, base form
        'VBD': 'VERB',      # 	VerbForm=fin Tense=past	                            verb, past tense
        'VBG': 'VERB',      # 	VerbForm=part Tense=pres Aspect=prog	            verb, gerund or present participle
        'VBN': 'VERB',      # 	VerbForm=part Tense=past Aspect=perf	            verb, past participle
        'VBP': 'VERB',      # 	VerbForm=fin Tense=pres	                            verb, non-3rd person singular present
        'VBZ': 'VERB',      # 	VerbForm=fin Tense=pres Number=sing Person=three	verb, 3rd person singular present
        'WDT': 'DET',       # 		                                                wh-determiner
        'WP': 'PRON',       # 		                                                wh-pronoun, personal
        'WP$': 'DET',       # 	Poss=yes	                                        wh-pronoun, possessive
        'WRB': 'ADV',       # 		                                                wh-adverb
        'XX': 'X',          # 		                                                unknown
        '_SP': 'SPACE'
    }

    @classmethod
    def response_key(cls) -> str:
        return 'spos_conll'

    @classmethod
    def num_classes(cls) -> int:
        return 16

    @classmethod
    def num_spans(cls) -> int:
        return 1

    @classmethod
    def simplify_pos(cls, pos: str) -> str:
        return cls._pos_map[pos]

    def _load(self, example_manager: CorpusExampleUnifier, use_meta_train: bool) -> RawData:
        return _load_probing_task(
            example_manager, self.path, type(self).response_key(), type(self).num_classes(), type(self).simplify_pos,
            use_meta_train=use_meta_train)


@dataclass(frozen=True)
class ConstituentsConll2012(CorpusBase):
    path: str = path_attribute_field('constituents_conll_2012_path')

    @classmethod
    def response_key(cls) -> str:
        return 'const_conll'

    @classmethod
    def num_classes(cls) -> int:
        return 30

    @classmethod
    def num_spans(cls) -> int:
        return 1

    def _load(self, example_manager: CorpusExampleUnifier, use_meta_train: bool) -> RawData:
        return _load_probing_task(
            example_manager, self.path, type(self).response_key(), type(self).num_classes(),
            use_meta_train=use_meta_train)


@dataclass(frozen=True)
class SemanticRoleLabelConll2012(CorpusBase):
    path: str = path_attribute_field('semantic_role_label_conll_2012_path')

    @classmethod
    def response_key(cls) -> str:
        return 'srl_conll'

    @classmethod
    def num_classes(cls) -> int:
        return 66

    @classmethod
    def num_spans(cls) -> int:
        return 2

    def _load(self, example_manager: CorpusExampleUnifier, use_meta_train: bool) -> RawData:
        return _load_probing_task(
            example_manager, self.path, type(self).response_key(), type(self).num_classes(),
            use_meta_train=use_meta_train)


@dataclass(frozen=True)
class NamedEntityRecognitionConll2012(CorpusBase):
    path: str = path_attribute_field('named_entity_recognition_conll_2012_path')

    @classmethod
    def response_key(cls) -> str:
        return 'ner_conll'

    @classmethod
    def num_classes(cls) -> int:
        return 18

    @classmethod
    def num_spans(cls) -> int:
        return 1

    def _load(self, example_manager: CorpusExampleUnifier, use_meta_train: bool) -> RawData:
        return _load_probing_task(
            example_manager, self.path, type(self).response_key(), type(self).num_classes(),
            use_meta_train=use_meta_train)


@dataclass(frozen=True)
class CoreferenceResolutionConll2012(CorpusBase):
    path: str = path_attribute_field('coreference_conll_2012_path')

    @classmethod
    def response_key(cls) -> str:
        return 'coref_conll'

    @classmethod
    def num_classes(cls) -> int:
        return 2

    @classmethod
    def num_spans(cls) -> int:
        return 2

    def _load(self, example_manager: CorpusExampleUnifier, use_meta_train: bool) -> RawData:
        return _load_probing_task(
            example_manager, self.path, type(self).response_key(), type(self).num_classes(),
            use_meta_train=use_meta_train)


@dataclass(frozen=True)
class DependenciesEnglishWeb(CorpusBase):
    path: str = path_attribute_field('dependencies_english_web_path')

    @classmethod
    def response_key(cls) -> str:
        return 'dep_ewt'  # ewt = English Web Treebank

    @classmethod
    def num_classes(cls) -> int:
        return 49

    @classmethod
    def num_spans(cls) -> int:
        return 2

    def _load(self, example_manager: CorpusExampleUnifier, use_meta_train: bool) -> RawData:
        return _load_probing_task(
            example_manager,
            os.path.join(self.path, 'en_ewt-ud-train.json'),
            type(self).response_key(),
            type(self).num_classes(),
            validation_path=os.path.join(self.path, 'en_ewt-ud-dev.json'),
            test_path=os.path.join(self.path, 'en_ewt-ud-test.json'),
            use_meta_train=use_meta_train)


@dataclass(frozen=True)
class DefinitePronounResolution(CorpusBase):
    path: str = path_attribute_field('definite_pronoun_resolution_path')

    @classmethod
    def response_key(cls) -> str:
        return 'dpr'

    @classmethod
    def num_classes(cls) -> int:
        return 2

    @classmethod
    def num_spans(cls) -> int:
        return 2

    def _load(self, example_manager: CorpusExampleUnifier, use_meta_train: bool) -> RawData:
        return _load_probing_task(
            example_manager,
            os.path.join(self.path, 'train.json'),
            type(self).response_key(),
            type(self).num_classes(),
            validation_path=os.path.join(self.path, 'dev.json'),
            test_path=os.path.join(self.path, 'test.json'),
            use_meta_train=use_meta_train)


@dataclass(frozen=True)
class SemEval(CorpusBase):
    path: str = path_attribute_field('sem_eval_path')

    @classmethod
    def response_key(cls) -> str:
        return 'sem_eval'

    @classmethod
    def num_classes(cls) -> int:
        return 19

    @classmethod
    def num_spans(cls) -> int:
        return 2

    def _load(self, example_manager: CorpusExampleUnifier, use_meta_train: bool) -> RawData:
        return _load_probing_task(
            example_manager,
            os.path.join(self.path, 'train.0.85.json'),
            type(self).response_key(),
            type(self).num_classes(),
            validation_path=os.path.join(self.path, 'dev.json'),
            test_path=os.path.join(self.path, 'test.json'),
            use_meta_train=use_meta_train)


@dataclass(frozen=True)
class SemanticProtoRoles1(CorpusBase):
    path: str = path_attribute_field('semantic_proto_roles_1_path')

    @classmethod
    def num_classes(cls) -> int:
        return 2

    @classmethod
    def num_spans(cls) -> int:
        return 2

    def _load(self, example_manager: CorpusExampleUnifier, use_meta_train: bool) -> RawData:
        return _load_multi_binary_label_probing_task(
            example_manager,
            os.path.join(self.path, 'spr1.train.json'),
            'spr1',
            validation_path=os.path.join(self.path, 'spr1.dev.json'),
            test_path=os.path.join(self.path, 'spr1.test.json'),
            use_meta_train=use_meta_train)


@dataclass(frozen=True)
class SemanticProtoRoles2(CorpusBase):
    path: str = path_attribute_field('semantic_proto_roles_2_path')

    @classmethod
    def num_classes(cls) -> int:
        return 2

    @classmethod
    def num_spans(cls) -> int:
        return 2

    def _load(self, example_manager: CorpusExampleUnifier, use_meta_train: bool) -> RawData:
        return _load_multi_binary_label_probing_task(
            example_manager,
            os.path.join(self.path, 'edges.train.json'),
            'spr2',
            validation_path=os.path.join(self.path, 'edges.dev.json'),
            test_path=os.path.join(self.path, 'edges.test.json'),
            use_meta_train=use_meta_train)
