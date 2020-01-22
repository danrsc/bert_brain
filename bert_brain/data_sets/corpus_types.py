from .boolean_questions import BooleanQuestions
from .choice_of_plausible_alternatives import ChoiceOfPlausibleAlternatives
from .colorless_green import ColorlessGreenCorpus, LinzenAgreementCorpus
from .commitment_bank import CommitmentBank
from .harry_potter import HarryPotterCorpus
from .multi_sentence_reading_comprehension import MultiSentenceReadingComprehension
from .natural_stories import NaturalStoriesCorpus
from .reading_comprehension_with_common_sense_reasoning import ReadingComprehensionWithCommonSenseReasoning
from .recognizing_textual_entailment import RecognizingTextualEntailment
from .stanford_sentiment_treebank import StanfordSentimentTreebank
from .university_college_london_corpus import UclCorpus
from .winograd_schema_challenge import WinogradSchemaChallenge
from .word_in_context import WordInContext

__all__ = [
    'BooleanQuestions',
    'ChoiceOfPlausibleAlternatives',
    'ColorlessGreenCorpus',
    'LinzenAgreementCorpus',
    'CommitmentBank',
    'HarryPotterCorpus',
    'MultiSentenceReadingComprehension',
    'NaturalStoriesCorpus',
    'ReadingComprehensionWithCommonSenseReasoning',
    'RecognizingTextualEntailment',
    'StanfordSentimentTreebank',
    'UclCorpus',
    'WinogradSchemaChallenge',
    'WordInContext']
