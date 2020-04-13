from .boolean_questions import BooleanQuestions
from .choice_of_plausible_alternatives import ChoiceOfPlausibleAlternatives
from .colorless_green import ColorlessGreenCorpus, LinzenAgreementCorpus
from .commitment_bank import CommitmentBank
from .dundee import DundeeCorpus
from .ghent_eye_tracking_corpus import GhentEyeTrackingCorpus
from .harry_potter import HarryPotterCorpus
from .multi_sentence_reading_comprehension import MultiSentenceReadingComprehension
from .natural_stories import NaturalStoriesCorpus
from .reading_comprehension_with_common_sense_reasoning import ReadingComprehensionWithCommonSenseReasoning
from .recognizing_textual_entailment import RecognizingTextualEntailment
from .stanford_sentiment_treebank import StanfordSentimentTreebank
from .university_college_london_corpus import UclCorpus
from .winograd_schema_challenge import WinogradSchemaChallenge
from .word_in_context import WordInContext
from .what_you_can_cram import BigramShift, CoordinationInversion, ObjectNumber, SemanticOddManOut, SentenceLength, \
    SubjectNumber, TopConstituents, TreeDepth, VerbTense, WordContent
from .edge_probing import PartOfSpeechConll2012, SimplifiedPartOfSpeechConll2012, ConstituentsConll2012, \
    SemanticRoleLabelConll2012, NamedEntityRecognitionConll2012, CoreferenceResolutionConll2012, \
    DependenciesEnglishWeb, DefinitePronounResolution, SemEval, SemanticProtoRoles1, SemanticProtoRoles2


__all__ = [
    'BooleanQuestions',
    'ChoiceOfPlausibleAlternatives',
    'ColorlessGreenCorpus',
    'LinzenAgreementCorpus',
    'CommitmentBank',
    'DundeeCorpus',
    'GhentEyeTrackingCorpus',
    'HarryPotterCorpus',
    'MultiSentenceReadingComprehension',
    'NaturalStoriesCorpus',
    'ReadingComprehensionWithCommonSenseReasoning',
    'RecognizingTextualEntailment',
    'StanfordSentimentTreebank',
    'UclCorpus',
    'WinogradSchemaChallenge',
    'WordInContext',
    'BigramShift',
    'CoordinationInversion',
    'ObjectNumber',
    'SemanticOddManOut',
    'SentenceLength',
    'SubjectNumber',
    'TopConstituents',
    'TreeDepth',
    'VerbTense',
    'WordContent',
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
