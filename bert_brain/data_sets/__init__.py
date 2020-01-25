from . import syntactic_dependency
from . import boolean_questions
from . import choice_of_plausible_alternatives
from . import colorless_green
from . import commitment_bank
from . import corpus_base
from . import corpus_dataset_factory
from . import corpus_types
from . import data_id_dataset
from . import data_id_multidataset
from . import data_preparer
from . import fmri_example_builders
from . import harry_potter
from . import input_features
from . import multi_sentence_reading_comprehension
from . import natural_stories
from . import preprocessors
from . import reading_comprehension_with_common_sense_reasoning
from . import recognizing_textual_entailment
from . import spacy_token_meta
from . import stanford_sentiment_treebank
from . import university_college_london_corpus
from . import what_you_can_cram
from . import winograd_schema_challenge
from . import word_in_context

from .syntactic_dependency import *
from .boolean_questions import *
from .choice_of_plausible_alternatives import *
from .colorless_green import *
from .commitment_bank import *
from .corpus_base import *
from .corpus_dataset_factory import *
from .corpus_types import *
from .data_id_dataset import *
from .data_id_multidataset import *
from .data_preparer import *
from .fmri_example_builders import *
from .harry_potter import *
from .input_features import *
from .multi_sentence_reading_comprehension import *
from .natural_stories import *
from .preprocessors import *
from .reading_comprehension_with_common_sense_reasoning import *
from .recognizing_textual_entailment import *
from .spacy_token_meta import *
from .stanford_sentiment_treebank import *
from .university_college_london_corpus import *
from .what_you_can_cram import *
from .winograd_schema_challenge import *
from .word_in_context import *

__all__ = [
    'syntactic_dependency', 'boolean_questions', 'choice_of_plausible_alternatives', 'colorless_green',
    'commitment_bank', 'corpus_base', 'corpus_dataset_factory', 'corpus_types', 'data_id_dataset',
    'data_id_multidataset', 'data_preparer', 'fmri_example_builders', 'harry_potter', 'input_features',
    'multi_sentence_reading_comprehension', 'natural_stories', 'preprocessors',
    'reading_comprehension_with_common_sense_reasoning', 'recognizing_textual_entailment', 'spacy_token_meta',
    'stanford_sentiment_treebank', 'university_college_london_corpus', 'what_you_can_cram',
    'winograd_schema_challenge', 'word_in_context']
__all__.extend(syntactic_dependency.__all__)
__all__.extend(boolean_questions.__all__)
__all__.extend(choice_of_plausible_alternatives.__all__)
__all__.extend(colorless_green.__all__)
__all__.extend(commitment_bank.__all__)
__all__.extend(corpus_base.__all__)
__all__.extend(corpus_dataset_factory.__all__)
__all__.extend(data_id_dataset.__all__)
__all__.extend(data_id_multidataset.__all__)
__all__.extend(data_preparer.__all__)
__all__.extend(fmri_example_builders.__all__)
__all__.extend(harry_potter.__all__)
__all__.extend(input_features.__all__)
__all__.extend(multi_sentence_reading_comprehension.__all__)
__all__.extend(natural_stories.__all__)
__all__.extend(preprocessors.__all__)
__all__.extend(reading_comprehension_with_common_sense_reasoning.__all__)
__all__.extend(recognizing_textual_entailment.__all__)
__all__.extend(spacy_token_meta.__all__)
__all__.extend(stanford_sentiment_treebank.__all__)
__all__.extend(university_college_london_corpus.__all__)
__all__.extend(what_you_can_cram.__all__)
__all__.extend(winograd_schema_challenge.__all__)
__all__.extend(word_in_context.__all__)


def _assert_corpus_subclasses_recursive():
    def sub(c, result):
        result.add(c.__name__)
        for sc in c.__subclasses__():
            sub(sc, result)
    all_corpus_types = set()
    for cb in CorpusBase.__subclasses__():
        sub(cb, all_corpus_types)

    registered_corpus_types = set(corpus_types.__all__)
    unregistered_types = all_corpus_types - registered_corpus_types
    if len(unregistered_types) > 0:
        raise AssertionError('Some corpora are not registered in corpus_types: {}'.format(unregistered_types))


_assert_corpus_subclasses_recursive()
