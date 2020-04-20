import hashlib
import itertools
import random
from collections import OrderedDict
from dataclasses import replace, dataclass
from typing import Union, Iterable, Mapping, Tuple

import numpy as np
import torch
import torch.cuda

from transformers.modeling_bert import gelu_new as gelu

from .common import SwitchRemember
from .data_sets import ResponseKind, PreprocessDetrend, PreprocessStandardize, \
    PreprocessKMeans, PreprocessRandomPair, PreprocessMakeBinary, PreprocessForkNoClusterToDisk, \
    PreprocessQuantileDigitize, PreprocessPCA, PreprocessRankData, PreprocessLog, corpus_types, \
    BatchOneTaskSamplerFactory, BatchOneTaskRandomSamplerFactory, BatchOneTaskProportionalSamplerFactory, \
    BatchOneTaskTemperatureProportionalSamplerFactory, BatchOneTaskTaskPermutedSamplerFactory, \
    BatchOneTaskManualWeightSamplerFactory
from .modeling import KeyedLinear, LinearContextualParameterGeneration, PooledFromSequence, PooledFromKTokens, \
    MarkedTokenConcatFixedNumTokens, GroupMultipart, KeyedSingleTargetSpanAttention, critic_types, \
    learning_rate_schedules, MultiLayerBottleneck, ContextualBottleneckSum, AttentionKeyValues, AttentionPool, \
    ContextAttention, LinearDecreasingTemperatureSchedule, KeyedConcat, KeyedSingleTargetSpanMaxPool
from .settings import Settings, OptimizationSettings


__all__ = [
    'task_hash',
    'set_random_seeds',
    'iterate_powerset',
    'named_variations',
    'singleton_variation',
    'make_standard_head_graph']


def _internal_hash_update(hash_, settings: Settings):
    hash_.update('standard_losses'.encode())
    for loss_task in sorted(settings.loss_tasks):
        hash_.update(loss_task.encode())
    hash_.update('meta_learn_no_gradient_losses'.encode())
    for loss_task in sorted(settings.meta_learn_no_gradient_loss_tasks):
        hash_.update(loss_task.encode())
    hash_.update('meta_learn_gradient_losses'.encode())
    for loss_task in sorted(settings.meta_learn_gradient_loss_tasks):
        hash_.update(loss_task.encode())
    if settings.load_from is not None:
        hash_.update('load_from')
        (variation_name, _), load_from_settings = singleton_variation(settings.load_from)
        hash_.update(variation_name.encode())
        _internal_hash_update(hash_, load_from_settings)


def task_hash(settings: Settings):
    hash_ = hashlib.sha256()
    _internal_hash_update(hash_, settings)
    return hash_.hexdigest()


def set_random_seeds(seed, index_run, n_gpu):
    hash_ = hashlib.sha256('{}'.format(seed).encode())
    hash_.update('{}'.format(index_run).encode())
    seed = np.frombuffer(hash_.digest(), dtype='uint32')
    random_state = np.random.RandomState(seed)
    np.random.set_state(random_state.get_state())
    seed = np.random.randint(low=0, high=np.iinfo('uint32').max)
    random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed_all(seed)
    return seed


def rank_space(data):
    from scipy.stats.mstats import rankdata
    return rankdata(data, axis=0)


def iterate_powerset(items):
    for sub_set in itertools.chain.from_iterable(
            itertools.combinations(items, num) for num in range(1, len(items) + 1)):
        yield sub_set


def make_standard_head_graph(corpus, sequence_key=None, pooled_key=None):

    if pooled_key is None:
        pooled_key = ('bert', 'pooled')
    if sequence_key is None:
        sequence_key = ('bert', 'sequence')

    response_key = type(corpus).response_key() if hasattr(type(corpus), 'response_key') else None

    head = OrderedDict()

    if isinstance(corpus, corpus_types.NaturalStoriesCorpus):
        head[ResponseKind.ns_froi] = OrderedDict(
            froi_linear=KeyedLinear(
                pooled_key, apply_at_most_one_data_id='if_no_target',
                targets=ResponseKind.ns_froi))
        return head
    elif isinstance(corpus, corpus_types.HarryPotterCorpus):
        if corpus.meg_subjects is None or len(corpus.meg_subjects) > 0:
            head[ResponseKind.hp_meg] = OrderedDict(meg_linear=KeyedLinear(
                sequence_key, targets=ResponseKind.hp_meg))
        if corpus.fmri_subjects is None or len(corpus.fmri_subjects) > 0:
            head[ResponseKind.hp_fmri] = OrderedDict(fmri_linear=KeyedLinear(
                pooled_key,
                apply_at_most_one_data_id='if_no_target',
                targets=ResponseKind.hp_fmri))
        return head
    elif isinstance(corpus, corpus_types.WordInContext):
        head['{}_group'.format(response_key)] = MarkedTokenConcatFixedNumTokens(
            2,
            response_key, 'data_ids',
            '{}_concat'.format(response_key),
            sequence_key)
        head['{}_linear'.format(response_key)] = KeyedLinear(
            '{}_concat'.format(response_key),
            output_key_to_shape={response_key: 1}, apply_at_most_one_data_id=True)
        return head
    elif isinstance(
            corpus,
            (corpus_types.ChoiceOfPlausibleAlternatives,
             corpus_types.ReadingComprehensionWithCommonSenseReasoning,
             corpus_types.MultiSentenceReadingComprehension)):
        head['{}_linear'.format(response_key)] = KeyedLinear(
            pooled_key, output_key_to_shape={
                '{}_choice'.format(response_key): 1}, apply_at_most_one_data_id=True)
        head['{}_mc'.format(response_key)] = GroupMultipart(
            None, 'multipart_id', response_key, '{}_choice'.format(response_key))
        return head
    elif isinstance(corpus, corpus_types.WinogradSchemaChallenge):
        head['{}_span_linear'.format(response_key)] = KeyedSingleTargetSpanAttention(
            2, sequence_key, 'span_ids', conv_hidden_channels=1024, conv_hidden_kernel=1,
            output_key_to_shape={response_key: 1})
        return head
    else:
        return head


def standard_edge_probing_graph(corpora, sequence_key):
    num_spans_to_head = dict()
    result = OrderedDict()
    for corpus in corpora:
        if isinstance(  # edge probing
                corpus,
                (corpus_types.PartOfSpeechConll2012,
                 corpus_types.SimplifiedPartOfSpeechConll2012,
                 corpus_types.ConstituentsConll2012,
                 corpus_types.NamedEntityRecognitionConll2012,
                 corpus_types.SemanticRoleLabelConll2012,
                 corpus_types.CoreferenceResolutionConll2012,
                 corpus_types.DependenciesEnglishWeb,
                 corpus_types.DefinitePronounResolution,
                 corpus_types.SemEval,
                 corpus_types.SemanticProtoRoles1,
                 corpus_types.SemanticProtoRoles2)):
            inner_name = 'max_pool_linear_1_span' \
                if corpus.num_spans() == 1 else 'max_pool_linear_{}_spans'.format(corpus.num_spans())
            if corpus.num_spans() not in num_spans_to_head:
                num_spans_to_head[corpus.num_spans()] = OrderedDict()
                num_spans_to_head[corpus.num_spans()][inner_name] = KeyedSingleTargetSpanMaxPool(
                    corpus.num_spans(), sequence_key, 'span_ids')
            num_spans_to_head[corpus.num_spans()][inner_name].targets.add(corpus.corpus_key)
            result[corpus.corpus_key] = num_spans_to_head[corpus.num_spans()]
    return result


@dataclass(frozen=True)
class SamplerWeightCollapse:
    collapse_prefixes: Tuple[str, ...]
    divisor: int = 1
    temperature: int = 1

    def __call__(self, task_weights: Mapping[str, float]) -> Mapping[str, float]:
        prefix_maxes = dict((p, 0) for p in self.collapse_prefixes)
        prefix_counts = dict((p, 0) for p in self.collapse_prefixes)
        other = 0
        for key in task_weights:
            found = False
            for prefix in self.collapse_prefixes:
                if key.startswith(prefix):
                    prefix_maxes[prefix] = max(prefix_maxes[prefix], task_weights[key] / self.divisor)
                    prefix_counts[prefix] += 1
                    found = True
                    break
            if not found:
                other += task_weights[key]
        new_other = 1
        for prefix in prefix_maxes:
            new_other -= prefix_maxes[prefix] / prefix_counts[prefix]
        other_correction = new_other / other
        weights = list()
        for key in task_weights:
            found = False
            for prefix in self.collapse_prefixes:
                if key.startswith(prefix):
                    weights.append(prefix_maxes[prefix] / (prefix_counts[prefix] ** 2))
                    found = True
                    break
            if not found:
                weights.append(task_weights[key] * other_correction)
        weights = np.array(weights)
        weights = np.power(weights, 1 / self.temperature)
        weights = weights / np.sum(weights)
        return dict(zip(task_weights, weights))


def singleton_variation(name: Union[str, Tuple[str, int]]) -> Tuple[Tuple[str, str], Settings]:
    result = named_variations(name)
    if len(result) != 1:
        raise ValueError('More than one result for variation: {}'.format(name))
    for k in result:
        return k, result[k]


def named_variations(name: Union[str, Tuple[str, int]]) -> Mapping[Tuple[str, str], Settings]:
    if not isinstance(name, str):
        if len(name) != 2:
            raise ValueError('name must be either a string or a (string, int) tuple')
        name, name_index = name
        name_index = int(name_index)
        result = _named_variations(name)
        try:
            for i, sub in enumerate(result):
                if name_index == i:
                    return {(name, '{}'.format(name_index)): sub}
            raise KeyError('Unable to find variation: {}'.format((name, name_index)))
        except TypeError:
            raise KeyError('Cannot use {} to access a variation which is not iterable'.format((name, name_index)))

    result = _named_variations(name)
    if isinstance(result, Settings):
        return {(name, '{}'.format(0)): result}
    elif isinstance(result, dict):
        return type(result)(((name, k), result[k]) for k in result)
    else:
        return OrderedDict(((name, '{}'.format(i)), r) for i, r in enumerate(result))


def _named_variations(name: Union[str, Tuple[str, int]]) -> Union[Settings, Iterable[Settings], Mapping[str, Settings]]:

    hp_fmri_tasks = tuple('hp_fmri_{}'.format(s) for s in corpus_types.HarryPotterCorpus.all_fmri_subjects)

    name = SwitchRemember(name)

    if name == 'erp':
        return [
            Settings(corpora=(corpus_types.UclCorpus(),), num_runs=100, loss_tasks=set(t))
            for t in iterate_powerset(corpus_types.UclCorpus.all_erp_tasks)]
    elif name == 'erp_joint':
        return Settings(
            corpora=(corpus_types.UclCorpus(),), loss_tasks=set(corpus_types.UclCorpus.all_erp_tasks), num_runs=100)
    elif name == 'nat_stories':
        settings = Settings(
            corpora=(
                corpus_types.NaturalStoriesCorpus(
                    froi_window_duration=10.,
                    froi_minimum_duration_required=9.5,
                    froi_use_word_unit_durations=False,
                    froi_sentence_mode='ignore'),
                corpus_types.UclCorpus()),
            optimization_settings=OptimizationSettings(num_train_epochs=50),
            num_runs=10)
        settings.head_graph_parts[ResponseKind.ns_froi] = OrderedDict(
            untransformed_pooled_linear=KeyedLinear(
                ('bert', 'untransformed_pooled'), apply_at_most_one_data_id='if_no_target',
                targets=ResponseKind.ns_froi))
        return [replace(settings, loss_tasks=set(t)) for t in [
            ('ns_spr',),
            corpus_types.UclCorpus.all_erp_tasks + ('ns_spr',),
            corpus_types.NaturalStoriesCorpus.all_froi_tasks + ('ns_spr',),
            corpus_types.UclCorpus.all_erp_tasks + corpus_types.NaturalStoriesCorpus.all_froi_tasks,
            corpus_types.UclCorpus.all_erp_tasks + corpus_types.NaturalStoriesCorpus.all_froi_tasks + ('ns_spr',)]]
    elif name == 'ns_froi':
        settings = Settings(
            corpora=(
                corpus_types.NaturalStoriesCorpus(
                    froi_sentence_mode='ignore',
                    froi_window_duration=10.,
                    froi_minimum_duration_required=9.5,
                    froi_use_word_unit_durations=False,
                    froi_minimum_story_count=2,
                    include_reaction_times=False),),
            optimization_settings=OptimizationSettings(num_train_epochs=3),
            num_runs=7,
            loss_tasks=set(corpus_types.NaturalStoriesCorpus.all_froi_tasks))
        settings.head_graph_parts[ResponseKind.ns_froi] = OrderedDict(
            untransformed_pooled_linear=KeyedLinear(
                ('bert', 'untransformed_pooled'), apply_at_most_one_data_id='if_no_target',
                targets=ResponseKind.ns_froi))
        return settings
    elif name == 'number_agreement':
        agr = ('colorless', 'linzen_agree')
        return [
            Settings(
                corpora=(
                    corpus_types.ColorlessGreenCorpus(),
                    corpus_types.LinzenAgreementCorpus(),
                    corpus_types.UclCorpus()),
                optimization_settings=OptimizationSettings(num_train_epochs=50),
                num_runs=10,
                loss_tasks=set(t))
            for t in [agr, corpus_types.UclCorpus.all_erp_tasks + agr, corpus_types.UclCorpus.all_erp_tasks]]
    elif name == 'hp_fmri_I_20':
        settings = Settings(
            corpora=(corpus_types.HarryPotterCorpus(
                fmri_subjects=['I'],
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=False,
                meg_subjects=[]),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=20,
                num_epochs_train_prediction_heads_only=2,
                num_final_epochs_train_prediction_heads_only=0),
            loss_tasks={'hp_fmri_I'},
            num_runs=4)
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True)]
        settings.head_graph_parts[ResponseKind.hp_fmri] = OrderedDict(
            untransformed_pooled_linear=KeyedLinear(
                ('bert', 'untransformed_pooled'), apply_at_most_one_data_id='if_no_target',
                targets=ResponseKind.hp_fmri))
        return settings
    elif name == 'hp_fmri_I_reptile':
        settings = Settings(
            corpora=(corpus_types.HarryPotterCorpus(
                fmri_subjects=['I'],
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=False,
                meg_subjects=[]),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=2,
                num_epochs_train_prediction_heads_only=0,
                num_final_epochs_train_prediction_heads_only=0),
            meta_learn_gradient_loss_tasks={'hp_fmri_I'},
            num_meta_learn_gradient_samples=10,
            num_meta_learn_no_gradient_samples=0,
            sampler_factory=BatchOneTaskRandomSamplerFactory(),
            num_runs=4)
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True)]
        settings.head_graph_parts[ResponseKind.hp_fmri] = OrderedDict(
            untransformed_pooled_linear=KeyedLinear(
                ('bert', 'untransformed_pooled'), apply_at_most_one_data_id='if_no_target',
                targets=ResponseKind.hp_fmri))
        return settings
    elif name == 'hp_fmri_meg_joint':
        settings = Settings(
            corpora=(corpus_types.HarryPotterCorpus(
                fmri_subjects=None,
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='leila',
                meg_subjects=None),),  # None means everyone
            optimization_settings=OptimizationSettings(
                num_train_epochs=60,
                num_epochs_train_prediction_heads_only=10,
                num_final_epochs_train_prediction_heads_only=0),
            num_runs=4)
        settings.preprocessors[ResponseKind.hp_meg] = [
            PreprocessDetrend(
                stop_mode='content', metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(
                stop_mode='content', metadata_example_group_by='fmri_runs', train_on_all=True, average_axis=None)]
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True)]
        settings.head_graph_parts[ResponseKind.hp_fmri] = OrderedDict(
            untransformed_pooled_linear=KeyedLinear(
                ('bert', 'untransformed_pooled'), apply_at_most_one_data_id='if_no_target',
                targets=ResponseKind.hp_fmri, force_cpu=True))
        settings.head_graph_parts[ResponseKind.hp_meg] = OrderedDict(
            sequence_linear=KeyedLinear(
                ('bert', 'sequence'), targets=ResponseKind.hp_meg, force_cpu=True))
        settings.loss_tasks = set(hp_fmri_tasks + ('hp_meg',))
        return settings
    elif name in ['hp_fmri_simple_{}'.format(s) for s in corpus_types.HarryPotterCorpus.all_fmri_subjects]:
        subject_ = name[name.var.rindex('_') + 1:]
        settings = Settings(
            corpora=(corpus_types.HarryPotterCorpus(
                fmri_subjects=subject_,
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='leila',
                meg_subjects=[]),),  # None means everyone
            optimization_settings=OptimizationSettings(
                num_train_epochs=30,
                num_epochs_train_prediction_heads_only=10,
                num_final_epochs_train_prediction_heads_only=0),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg),
            num_runs=100,
            loss_tasks={'hp_fmri_{}'.format(subject_)})
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True)]
        settings.head_graph_parts[ResponseKind.hp_fmri] = OrderedDict(
            untransformed_pooled_linear=KeyedLinear(
                ('bert', 'untransformed_pooled'), apply_at_most_one_data_id='if_no_target',
                targets=ResponseKind.hp_fmri))
        return settings
    elif name == 'hp_fmri_simple':
        return OrderedDict(
            ('hp_fmri_simple_{}'.format(s), _named_variations('hp_fmri_simple_{}'.format(s)))
            for s in corpus_types.HarryPotterCorpus.all_fmri_subjects)
    elif name == 'sst':
        return Settings(corpora=(corpus_types.StanfordSentimentTreebank(),), loss_tasks={'sentiment'}, num_runs=1)
    elif name == 'hp_from_I':
        settings = Settings(
            corpora=(corpus_types.HarryPotterCorpus(
                fmri_subjects=None,
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                meg_subjects=[]),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=10,
                num_epochs_train_prediction_heads_only=-1),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg),
            num_runs=100)
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True)]
        settings.head_graph_parts[ResponseKind.hp_fmri] = OrderedDict(
            untransformed_pooled_linear=KeyedLinear(
                ('bert', 'untransformed_pooled'), apply_at_most_one_data_id='if_no_target',
                targets=ResponseKind.hp_fmri))
        training_variations = list()
        for s in corpus_types.HarryPotterCorpus.all_fmri_subjects:
            training_variations.append(
                replace(
                    settings,
                    loss_tasks={'hp_fmri_{}'.format(s)},
                    load_from='hp_fmri_I_20',
                    load_from_run_map=lambda run: run % 4))
            training_variations.append(replace(settings, loss_tasks={'hp_fmri_{}'.format(s)}))
        return training_variations
    elif name == 'hp_fmri_diff_cluster':
        settings = Settings(
            corpora=(corpus_types.HarryPotterCorpus(
                fmri_subjects=None,
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='leila',
                meg_subjects=[]),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=10,
                num_epochs_train_prediction_heads_only=1,
                num_final_epochs_train_prediction_heads_only=0),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg),
            num_runs=4,
            loss_tasks=set(hp_fmri_tasks))
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessKMeans(num_clusters=100, transform_fn=rank_space, n_init=100),
            ('diff', PreprocessRandomPair(
                num_samples_per_group=5000,
                metadata_example_group_by='fmri_runs',
                data_id_pair_fn_per_response_data=PreprocessRandomPair.pair_from_end)),
            PreprocessMakeBinary(threshold=0)]
        settings.preprocess_fork_fn = PreprocessForkNoClusterToDisk()
        settings.head_graph_parts[ResponseKind.hp_fmri] = OrderedDict(
            untransformed_pooled_linear=KeyedLinear(
                ('bert', 'untransformed_pooled'), apply_at_most_one_data_id='if_no_target',
                targets=ResponseKind.hp_fmri, hidden_sizes=[20]))
        settings.critics[ResponseKind.hp_fmri] = critic_types.NamedTargetSingleBinaryCrossEntropyWithLogits()
        return settings
    elif name == 'hp_meg_diff_drc_25':
        settings = Settings(
            corpora=(corpus_types.HarryPotterCorpus(
                fmri_subjects=[],
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='direct_rank_clustered_sum_25_ms',
                meg_subjects=None),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=100,
                num_epochs_train_prediction_heads_only=50,
                num_final_epochs_train_prediction_heads_only=0),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg),
            num_runs=4,
            loss_tasks=set('hp_meg_{}'.format(s) for s in corpus_types.HarryPotterCorpus.all_meg_subjects))
        settings.preprocessors[ResponseKind.hp_meg] = [
            PreprocessQuantileDigitize(
                quantiles=2,
                stop_mode='content',
                metadata_example_group_by='fmri_runs',
                train_on_all=True,
                use_one_hot=False)
        ]
        settings.critics[ResponseKind.hp_meg] = critic_types.NamedTargetStopWordAwareCrossEntropy(num_classes=2)
        return settings
    elif name == 'hp_meg_diff_drc_25_one':
        settings = Settings(
            corpora=(corpus_types.HarryPotterCorpus(
                fmri_subjects=[],
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='direct_rank_clustered_sum_25_ms',
                meg_subjects=None),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=100,
                num_epochs_train_prediction_heads_only=50,
                num_final_epochs_train_prediction_heads_only=0),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg),
            sampler_factory=BatchOneTaskSamplerFactory(100),
            weight_losses_by_inverse_example_counts=False,
            num_runs=4,
            loss_tasks=set('hp_meg_{}'.format(s) for s in corpus_types.HarryPotterCorpus.all_meg_subjects))
        settings.preprocessors[ResponseKind.hp_meg] = [
            PreprocessQuantileDigitize(
                quantiles=2,
                stop_mode='content',
                metadata_example_group_by='fmri_runs',
                train_on_all=True,
                use_one_hot=False)
        ]
        settings.critics[ResponseKind.hp_meg] = critic_types.NamedTargetStopWordAwareCrossEntropy(num_classes=2)
        return settings
    elif name == 'hp_meg_p75_drc_one':
        settings = Settings(
            corpora=(corpus_types.HarryPotterCorpus(
                fmri_subjects=[],
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='direct_rank_clustered_percentile_75_25_ms',
                meg_subjects=None),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=1000,
                num_epochs_train_prediction_heads_only=100,
                num_final_epochs_train_prediction_heads_only=0),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg),
            weight_losses_by_inverse_example_counts=True,
            num_runs=4,
            loss_tasks=set('hp_meg_{}'.format(s) for s in corpus_types.HarryPotterCorpus.all_meg_subjects))
        settings.preprocessors[ResponseKind.hp_meg] = [
            PreprocessQuantileDigitize(
                quantiles=2,
                stop_mode='content',
                metadata_example_group_by='fmri_runs',
                train_on_all=True,
                use_one_hot=False)
        ]

        settings.head_graph_parts[ResponseKind.hp_meg] = OrderedDict(meg_linear=KeyedLinear(
                ('bert', 'sequence', 'all'),
                hidden_sizes=[100], hidden_activation=None, targets=ResponseKind.hp_meg))

        settings.critics[ResponseKind.hp_meg] = critic_types.NamedTargetStopWordAwareCrossEntropy(num_classes=2)
        return settings
    elif name == 'hp_meg_cpg':
        settings = Settings(
            corpora=(corpus_types.HarryPotterCorpus(
                fmri_subjects=[],
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='direct_rank_clustered_percentile_75_25_ms',
                meg_subjects=None),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=1,
                num_epochs_train_prediction_heads_only=0,
                num_final_epochs_train_prediction_heads_only=0),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg),
            sampler_factory=BatchOneTaskSamplerFactory(100),
            weight_losses_by_inverse_example_counts=False,
            loss_tasks=set('hp_meg_{}'.format(s) for s in corpus_types.HarryPotterCorpus.all_meg_subjects),
            num_runs=4)
        settings.preprocessors[ResponseKind.hp_meg] = [
            PreprocessQuantileDigitize(
                quantiles=2,
                stop_mode='content',
                metadata_example_group_by='fmri_runs',
                train_on_all=True,
                use_one_hot=False)
        ]
        settings.common_graph_parts = OrderedDict(contextual_bottleneck=LinearContextualParameterGeneration(
            'response_id', 'num_response_data_fields', 3,
            OrderedDict(
                bottleneck=KeyedLinear(
                    ('bert', 'sequence', 'all'),
                    output_key_to_shape=OrderedDict(sequence_all_bottleneck=10),
                    should_norm=True))))

        settings.head_graph_parts[ResponseKind.hp_meg] = OrderedDict(meg_linear=KeyedLinear(
            'sequence_all_bottleneck', targets=ResponseKind.hp_meg))

        settings.critics[ResponseKind.hp_meg] = critic_types.NamedTargetStopWordAwareCrossEntropy(num_classes=2)
        return settings
    elif name == 'hp_fmri_reptile':
        settings = Settings(
            corpora=(corpus_types.HarryPotterCorpus(
                fmri_subjects=None,  # None means all
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                fmri_kind='rank_clustered',
                fmri_smooth_factor=None,
                group_meg_sentences_like_fmri=True,
                meg_subjects=[],
                meg_kind='direct_rank_clustered_percentile_75_25_ms'),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=2,
                num_epochs_train_prediction_heads_only=0,
                num_final_epochs_train_prediction_heads_only=0),
            meta_learn_gradient_loss_tasks=set(
                'hp_fmri_{}'.format(s) for s in corpus_types.HarryPotterCorpus.all_fmri_subjects),
            num_meta_learn_gradient_samples=10,
            num_meta_learn_no_gradient_samples=0,
            weight_losses_by_inverse_example_counts=False,
            sampler_factory=BatchOneTaskSamplerFactory(100),
            num_runs=4)
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessQuantileDigitize(
                quantiles=2,
                stop_mode=None,
                metadata_example_group_by='fmri_runs',
                train_on_all=True,
                use_one_hot=False)]
        settings.head_graph_parts[ResponseKind.hp_fmri] = OrderedDict(
            untransformed_pooled_linear=KeyedLinear(
                ('bert', 'untransformed_pooled'), apply_at_most_one_data_id='if_no_target',
                targets=ResponseKind.hp_fmri))
        settings.critics[ResponseKind.hp_fmri] = critic_types.NamedTargetSingleCrossEntropy(num_classes=2)
        return settings
    elif name == 'hp_fmri_meg_meta':
        settings = Settings(
            corpora=(corpus_types.HarryPotterCorpus(
                fmri_subjects=None,  # None means all
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                fmri_kind='rank_clustered',
                fmri_smooth_factor=None,
                group_meg_sentences_like_fmri=True,
                meg_subjects=None,  # None means all
                meg_kind='direct_rank_clustered_percentile_75_25_ms'),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=20,
                num_epochs_train_prediction_heads_only=0,
                num_final_epochs_train_prediction_heads_only=0),
            # loss_tasks=set('hp_fmri_{}'.format(s) for s in hp_fmri_subjects),
            meta_learn_gradient_loss_tasks=set(
                'hp_fmri_{}'.format(s) for s in corpus_types.HarryPotterCorpus.all_fmri_subjects).union(
                'hp_meg_{}'.format(s) for s in corpus_types.HarryPotterCorpus.all_meg_subjects),
            num_meta_learn_gradient_samples=10,
            num_meta_learn_no_gradient_samples=0,
            weight_losses_by_inverse_example_counts=False,
            sampler_factory=BatchOneTaskSamplerFactory(100),
            num_runs=4)
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessQuantileDigitize(
                quantiles=2,
                stop_mode=None,
                metadata_example_group_by='fmri_runs',
                train_on_all=True,
                use_one_hot=False)]
        settings.preprocessors[ResponseKind.hp_meg] = [
            PreprocessQuantileDigitize(
                quantiles=2,
                stop_mode='content',
                metadata_example_group_by='fmri_runs',
                train_on_all=True,
                use_one_hot=False)]
        # settings.common_graph_parts = OrderedDict(
        #     bottleneck=KeyedLinear(
        #         ('bert', 'sequence', 'all'), is_sequence=True,
        #         output_key_to_shape=OrderedDict(sequence_all_bottleneck=10),
        #         should_norm=True))

        settings.common_graph_parts = OrderedDict(
            contextual_bottleneck=LinearContextualParameterGeneration(
                'response_id', 'num_response_data_fields', 3,
                OrderedDict(
                    bottleneck=KeyedLinear(
                        ('bert', 'sequence', 'all'),
                        output_key_to_shape=OrderedDict(sequence_all_bottleneck=10),
                        should_norm=True))),
            pooled_bottleneck=PooledFromSequence('sequence_all_bottleneck', 'pooled_all_bottleneck'))
        settings.head_graph_parts[ResponseKind.hp_meg] = OrderedDict(meg_linear=KeyedLinear(
            'sequence_all_bottleneck', targets=ResponseKind.hp_meg))
        settings.head_graph_parts[ResponseKind.hp_fmri] = OrderedDict(fmri_linear=KeyedLinear(
            'pooled_all_bottleneck',
            apply_at_most_one_data_id='if_no_target',
            targets=ResponseKind.hp_fmri))
        settings.critics[ResponseKind.hp_meg] = critic_types.NamedTargetStopWordAwareCrossEntropy(num_classes=2)
        settings.critics[ResponseKind.hp_fmri] = critic_types.NamedTargetSingleCrossEntropy(num_classes=2)
        return settings
    elif name == 'superglue':
        settings = Settings(
            corpora=(
                corpus_types.BooleanQuestions(),
                corpus_types.CommitmentBank(),
                corpus_types.ChoiceOfPlausibleAlternatives(),
                corpus_types.MultiSentenceReadingComprehension(),
                corpus_types.ReadingComprehensionWithCommonSenseReasoning(),
                corpus_types.RecognizingTextualEntailment(),
                corpus_types.WinogradSchemaChallenge(),
                corpus_types.WordInContext()),
            max_sequence_length=256,
            optimization_settings=OptimizationSettings(
                num_train_epochs=10,
                num_epochs_train_prediction_heads_only=0,
                num_final_epochs_train_prediction_heads_only=0,
                learning_rate=1e-5,
                learning_rate_schedule=learning_rate_schedules.LinearWarmupSqrtDecayLearningRateScheduleFactory(400),
                train_batch_size=8,
                predict_batch_size=8),
            loss_tasks=set(),
            weight_losses_by_inverse_example_counts=False,
            sampler_factory=BatchOneTaskSamplerFactory(5000),
            num_runs=1)
        for corpus in settings.corpora:
            heads = make_standard_head_graph(
                corpus, sequence_key=('bert', 'sequence'), pooled_key=('bert', 'pooled'))
            settings.head_graph_parts.update(heads)
            settings.loss_tasks.add(type(corpus).response_key())
        return settings
    elif name == 'superglue_meta_cpg':
        settings = Settings(
            corpora=(
                corpus_types.BooleanQuestions(),
                corpus_types.CommitmentBank(),
                corpus_types.ChoiceOfPlausibleAlternatives(),
                corpus_types.MultiSentenceReadingComprehension(),
                corpus_types.ReadingComprehensionWithCommonSenseReasoning(),
                corpus_types.RecognizingTextualEntailment(),
                corpus_types.WinogradSchemaChallenge(),
                corpus_types.WordInContext()),
            max_sequence_length=256,
            optimization_settings=OptimizationSettings(
                num_train_epochs=10,
                num_epochs_train_prediction_heads_only=0,
                num_final_epochs_train_prediction_heads_only=0,
                learning_rate=1e-5,
                learning_rate_schedule=learning_rate_schedules.LinearWarmupSqrtDecayLearningRateScheduleFactory(400),
                train_batch_size=8,
                predict_batch_size=8),
            loss_tasks=set(),
            weight_losses_by_inverse_example_counts=False,
            num_meta_learn_gradient_samples=10,
            num_meta_learn_no_gradient_samples=0,
            sampler_factory=BatchOneTaskSamplerFactory(500),
            num_runs=1)
        settings.common_graph_parts = OrderedDict(
            contextual_bottleneck=LinearContextualParameterGeneration(
                'response_id', 'num_response_data_fields', 3,
                OrderedDict(
                    bottleneck=KeyedLinear(
                        ('bert', 'sequence', 'all'),
                        output_key_to_shape=OrderedDict(sequence_all_bottleneck=10),
                        should_norm=True))),
            pooled_bottleneck=PooledFromSequence('sequence_all_bottleneck', 'pooled_all_bottleneck'))
        for corpus in settings.corpora:
            heads = make_standard_head_graph(
                corpus, sequence_key='sequence_all_bottleneck', pooled_key='pooled_all_bottleneck')
            settings.head_graph_parts.update(heads)
            settings.meta_learn_gradient_loss_tasks.add(type(corpus).response_key())
        return settings
    elif name == 'cram_cpg':
        settings = Settings(
            corpora=(
                corpus_types.BigramShift(),
                corpus_types.CoordinationInversion(),
                corpus_types.ObjectNumber(),
                corpus_types.SemanticOddManOut(),
                corpus_types.SentenceLength(),
                corpus_types.SubjectNumber(),
                corpus_types.TopConstituents(),
                corpus_types.TreeDepth(),
                corpus_types.VerbTense(),
                corpus_types.WordContent()),
            optimization_settings=OptimizationSettings(
                num_train_epochs=10,
                num_epochs_train_prediction_heads_only=0,
                num_final_epochs_train_prediction_heads_only=0,
                learning_rate=1e-5,
                learning_rate_schedule=learning_rate_schedules.LinearWarmupSqrtDecayLearningRateScheduleFactory(400),
                train_batch_size=8,
                predict_batch_size=8),
            loss_tasks=set(),
            weight_losses_by_inverse_example_counts=False,
            num_meta_learn_gradient_samples=10,
            num_meta_learn_no_gradient_samples=0,
            sampler_factory=BatchOneTaskSamplerFactory(500),
            num_runs=1)
        settings.common_graph_parts = OrderedDict(
            contextual_bottleneck=LinearContextualParameterGeneration(
                'response_id', 'num_response_data_fields', 3,
                OrderedDict(
                    bottleneck=KeyedLinear(
                        ('bert', 'sequence', 'all'),
                        output_key_to_shape=OrderedDict(sequence_all_bottleneck=10),
                        should_norm=True))),
            pooled_bottleneck=PooledFromSequence('sequence_all_bottleneck', 'pooled_all_bottleneck'))
        for corpus in settings.corpora:
            heads = make_standard_head_graph(
                corpus, pooled_key='pooled_all_bottleneck', sequence_key='sequence_all_bottleneck')
            settings.head_graph_parts.update(heads)
            settings.meta_learn_gradient_loss_tasks.add(type(corpus).response_key())
        return settings
    elif name == 'fmri_cram_cpg':
        settings = Settings(
            corpora=(
                corpus_types.HarryPotterCorpus(
                    fmri_subjects=None,  # None means all
                    fmri_sentence_mode='ignore',
                    fmri_window_duration=10.1,
                    fmri_minimum_duration_required=9.6,
                    fmri_kind='rank_clustered',
                    fmri_smooth_factor=None,
                    separate_fmri_components=True,
                    group_meg_sentences_like_fmri=True,
                    meg_subjects=[]),
                corpus_types.BigramShift(),
                corpus_types.CoordinationInversion(),
                corpus_types.ObjectNumber(),
                corpus_types.SemanticOddManOut(),
                corpus_types.SentenceLength(),
                corpus_types.SubjectNumber(),
                corpus_types.TopConstituents(),
                corpus_types.TreeDepth(),
                corpus_types.VerbTense(),
                corpus_types.WordContent()),
            optimization_settings=OptimizationSettings(
                num_train_epochs=10,
                num_epochs_train_prediction_heads_only=0,
                num_final_epochs_train_prediction_heads_only=0,
                learning_rate=1e-5,
                learning_rate_schedule=learning_rate_schedules.LinearWarmupSqrtDecayLearningRateScheduleFactory(400),
                train_batch_size=8,
                predict_batch_size=8,
                num_loader_workers=1),
            loss_tasks=set(),
            data_id_in_batch_keys=None,
            field_spec_replacers={corpus_types.HarryPotterCorpus.__name__: {'is_sequence': False}},
            weight_losses_by_inverse_example_counts=False,
            num_meta_learn_gradient_samples=10,
            num_meta_learn_no_gradient_samples=0,
            sampler_factory=BatchOneTaskSamplerFactory(500),
            num_runs=1)
        settings.common_graph_parts = OrderedDict(
            contextual_bottleneck=LinearContextualParameterGeneration(
                'response_id', 'num_response_data_fields', 3,
                OrderedDict(
                    bottleneck=KeyedLinear(
                        ('bert', 'sequence', 'all'),
                        output_key_to_shape=OrderedDict(sequence_all_bottleneck=10),
                        should_norm=True))),
            pooled_bottleneck=PooledFromSequence('sequence_all_bottleneck', 'pooled_all_bottleneck'))
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessQuantileDigitize(
                quantiles=2,
                stop_mode=None,
                metadata_example_group_by='fmri_runs',
                train_on_all=True,
                use_one_hot=False)]
        settings.default_pooled_source = 'pooled_all_bottleneck'
        settings.default_sequence_source = 'sequence_all_bottleneck'
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.generic)
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.hp_fmri)
        return settings
    elif name == 'fmri_cram_cpg_prop':
        settings = Settings(
            corpora=(
                corpus_types.HarryPotterCorpus(
                    fmri_subjects=None,  # None means all
                    fmri_sentence_mode='ignore',
                    fmri_window_duration=10.1,
                    fmri_minimum_duration_required=9.6,
                    fmri_kind='rank_clustered',
                    fmri_smooth_factor=None,
                    separate_fmri_components=True,
                    group_meg_sentences_like_fmri=True,
                    meg_subjects=[]),
                corpus_types.BigramShift(),
                corpus_types.CoordinationInversion(),
                corpus_types.ObjectNumber(),
                corpus_types.SemanticOddManOut(),
                corpus_types.SentenceLength(),
                corpus_types.SubjectNumber(),
                corpus_types.TopConstituents(),
                corpus_types.TreeDepth(),
                corpus_types.VerbTense(),
                corpus_types.WordContent()),
            optimization_settings=OptimizationSettings(
                num_train_epochs=10,
                num_epochs_train_prediction_heads_only=0,
                num_final_epochs_train_prediction_heads_only=0,
                learning_rate_head=1e-3,
                learning_rate=1e-5,
                learning_rate_schedule=learning_rate_schedules.LinearWarmupSqrtDecayLearningRateScheduleFactory(1000),
                train_batch_size=8,
                predict_batch_size=8,
                num_loader_workers=8),
            loss_tasks=set(),
            data_id_in_batch_keys=None,
            field_spec_replacers={corpus_types.HarryPotterCorpus.__name__: {'is_sequence': False}},
            weight_losses_by_inverse_example_counts=False,
            num_meta_learn_gradient_samples=10,
            num_meta_learn_no_gradient_samples=0,
            sampler_factory=BatchOneTaskProportionalSamplerFactory(500),
            num_runs=4)
        settings.common_graph_parts = OrderedDict(
            contextual_bottleneck=LinearContextualParameterGeneration(
                'response_id', 'num_response_data_fields', 20,
                OrderedDict(
                    bottleneck=KeyedLinear(
                        ('bert', 'sequence', 'all'),
                        output_key_to_shape=OrderedDict(sequence_all_bottleneck=10),
                        should_norm=True))),
            pooled_bottleneck=PooledFromSequence('sequence_all_bottleneck', 'pooled_all_bottleneck'))
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessQuantileDigitize(
                quantiles=2,
                stop_mode=None,
                metadata_example_group_by='fmri_runs',
                train_on_all=True,
                use_one_hot=False)]
        settings.critics[ResponseKind.hp_fmri] = critic_types.NamedTargetSingleBinaryCrossEntropyWithLogits()
        settings.default_pooled_source = 'pooled_all_bottleneck'
        settings.default_sequence_source = 'sequence_all_bottleneck'
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.generic)
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.hp_fmri)
        return settings
    elif name == 'fmri_cram_cpg_prop_mlb':
        settings = Settings(
            corpora=(
                corpus_types.HarryPotterCorpus(
                    fmri_subjects=None,  # None means all
                    fmri_sentence_mode='ignore',
                    fmri_window_duration=10.1,
                    fmri_minimum_duration_required=9.6,
                    fmri_kind='rank_clustered',
                    fmri_smooth_factor=None,
                    separate_fmri_components=True,
                    group_meg_sentences_like_fmri=True,
                    meg_subjects=[]),
                corpus_types.BigramShift(),
                corpus_types.CoordinationInversion(),
                corpus_types.ObjectNumber(),
                corpus_types.SemanticOddManOut(),
                corpus_types.SentenceLength(),
                corpus_types.SubjectNumber(),
                corpus_types.TopConstituents(),
                corpus_types.TreeDepth(),
                corpus_types.VerbTense(),
                corpus_types.WordContent()),
            optimization_settings=OptimizationSettings(
                num_train_epochs=10,
                num_epochs_train_prediction_heads_only=0,
                num_final_epochs_train_prediction_heads_only=0,
                learning_rate_head=1e-3,
                learning_rate=1e-5,
                learning_rate_schedule=learning_rate_schedules.LinearWarmupSqrtDecayLearningRateScheduleFactory(1000),
                train_batch_size=8,
                predict_batch_size=8,
                num_loader_workers=8),
            loss_tasks=set(),
            data_id_in_batch_keys=None,
            field_spec_replacers={corpus_types.HarryPotterCorpus.__name__: {'is_sequence': False}},
            weight_losses_by_inverse_example_counts=False,
            num_meta_learn_gradient_samples=10,
            num_meta_learn_no_gradient_samples=0,
            sampler_factory=BatchOneTaskProportionalSamplerFactory(500),
            num_runs=4)
        settings.common_graph_parts = OrderedDict(
            multi_layer_bottleneck=MultiLayerBottleneck(
                ('bert', 'sequence', 'all'),
                output_name='multi_layer_sequence_bottleneck',
                num_bottleneck_channels=10,
                num_output_bottlenecks=3,
                num_hidden_bottlenecks=[100],
                hidden_activation=None,
                should_norm_hidden=True,
                should_transpose_output=False,
                should_norm=True),
            contextual_bottleneck=ContextualBottleneckSum(
                'response_id',
                'num_response_data_fields',
                'multi_layer_sequence_bottleneck',
                'sequence_all_bottleneck'),
            pooled_bottleneck=PooledFromSequence('sequence_all_bottleneck', 'pooled_all_bottleneck'))
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessQuantileDigitize(
                quantiles=2,
                stop_mode=None,
                metadata_example_group_by='fmri_runs',
                train_on_all=True,
                use_one_hot=False)]
        settings.critics[ResponseKind.hp_fmri] = critic_types.NamedTargetSingleBinaryCrossEntropyWithLogits()
        settings.default_pooled_source = 'pooled_all_bottleneck'
        settings.default_sequence_source = 'sequence_all_bottleneck'
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.generic)
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.hp_fmri)
        return settings
    elif name == 'fmri_cram_cpg_prop_wide':
        settings = Settings(
            corpora=(
                corpus_types.HarryPotterCorpus(
                    fmri_subjects=None,  # None means all
                    fmri_sentence_mode='ignore',
                    fmri_window_duration=10.1,
                    fmri_minimum_duration_required=9.6,
                    fmri_kind='rank_clustered',
                    fmri_smooth_factor=None,
                    separate_fmri_components=True,
                    group_meg_sentences_like_fmri=True,
                    meg_subjects=[]),
                corpus_types.BigramShift(),
                corpus_types.CoordinationInversion(),
                corpus_types.ObjectNumber(),
                corpus_types.SemanticOddManOut(),
                corpus_types.SentenceLength(),
                corpus_types.SubjectNumber(),
                corpus_types.TopConstituents(),
                corpus_types.TreeDepth(),
                corpus_types.VerbTense(),
                corpus_types.WordContent()),
            optimization_settings=OptimizationSettings(
                num_train_epochs=10,
                num_epochs_train_prediction_heads_only=0,
                num_final_epochs_train_prediction_heads_only=0,
                learning_rate_head=1e-3,
                learning_rate=1e-5,
                learning_rate_schedule=learning_rate_schedules.LinearWarmupSqrtDecayLearningRateScheduleFactory(1000),
                train_batch_size=8,
                predict_batch_size=8,
                num_loader_workers=8),
            loss_tasks=set(),
            data_id_in_batch_keys=None,
            field_spec_replacers={corpus_types.HarryPotterCorpus.__name__: {'is_sequence': False}},
            weight_losses_by_inverse_example_counts=False,
            num_meta_learn_gradient_samples=10,
            num_meta_learn_no_gradient_samples=0,
            sampler_factory=BatchOneTaskProportionalSamplerFactory(500),
            num_runs=4)
        settings.common_graph_parts = OrderedDict(
            multi_layer_bottleneck=MultiLayerBottleneck(
                ('bert', 'sequence', 'all'),
                output_name='multi_layer_sequence_bottleneck',
                num_bottleneck_channels=1,
                num_output_bottlenecks=100,
                num_hidden_bottlenecks=None,
                hidden_activation=None,
                should_norm_hidden=True,
                should_transpose_output=False,
                should_norm=False),
            contextual_bottleneck=ContextualBottleneckSum(
                'response_id',
                'num_response_data_fields',
                'multi_layer_sequence_bottleneck',
                'sequence_all_bottleneck'),
            pooled_bottleneck=PooledFromSequence('sequence_all_bottleneck', 'pooled_all_bottleneck'))
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessQuantileDigitize(
                quantiles=2,
                stop_mode=None,
                metadata_example_group_by='fmri_runs',
                train_on_all=True,
                use_one_hot=False)]
        settings.critics[ResponseKind.hp_fmri] = critic_types.NamedTargetSingleBinaryCrossEntropyWithLogits()
        settings.default_pooled_source = 'pooled_all_bottleneck'
        settings.default_sequence_source = 'sequence_all_bottleneck'
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.generic)
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.hp_fmri)
        return settings
    elif name == 'fmri_cram_cpg_prop_20_wide':
        settings = Settings(
            corpora=(
                corpus_types.HarryPotterCorpus(
                    fmri_subjects=None,  # None means all
                    fmri_sentence_mode='ignore',
                    fmri_window_duration=10.1,
                    fmri_minimum_duration_required=9.6,
                    fmri_kind='rank_clustered',
                    fmri_smooth_factor=None,
                    separate_fmri_components=True,
                    group_meg_sentences_like_fmri=True,
                    meg_subjects=[]),
                corpus_types.BigramShift(),
                corpus_types.CoordinationInversion(),
                corpus_types.ObjectNumber(),
                corpus_types.SemanticOddManOut(),
                corpus_types.SentenceLength(),
                corpus_types.SubjectNumber(),
                corpus_types.TopConstituents(),
                corpus_types.TreeDepth(),
                corpus_types.VerbTense(),
                corpus_types.WordContent()),
            optimization_settings=OptimizationSettings(
                num_train_epochs=20,
                num_epochs_train_prediction_heads_only=0,
                num_final_epochs_train_prediction_heads_only=0,
                # learning_rate_head=1e-3,
                learning_rate=1e-4,
                learning_rate_schedule=learning_rate_schedules.LinearWarmupSqrtDecayLearningRateScheduleFactory(1000),
                train_batch_size=8,
                predict_batch_size=8,
                num_loader_workers=8),
            loss_tasks=set(),
            data_id_in_batch_keys=None,
            field_spec_replacers={corpus_types.HarryPotterCorpus.__name__: {'is_sequence': False}},
            weight_losses_by_inverse_example_counts=False,
            num_meta_learn_gradient_samples=10,
            num_meta_learn_no_gradient_samples=0,
            sampler_factory=BatchOneTaskProportionalSamplerFactory(500),
            num_runs=4)
        settings.common_graph_parts = OrderedDict(
            bottleneck=KeyedLinear(
                ('bert', 'sequence', 'all'), output_key_to_shape={'sequence_all_bottleneck': 10}, should_norm=True),
            pooled_bottleneck=PooledFromSequence('sequence_all_bottleneck', 'pooled_all_bottleneck'))
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessQuantileDigitize(
                quantiles=2,
                stop_mode=None,
                metadata_example_group_by='fmri_runs',
                train_on_all=True,
                use_one_hot=False)]
        settings.critics[ResponseKind.hp_fmri] = critic_types.NamedTargetSingleBinaryCrossEntropyWithLogits()
        settings.default_pooled_source = 'pooled_all_bottleneck'
        settings.default_sequence_source = 'sequence_all_bottleneck'
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.generic)
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.hp_fmri)
        return settings
    elif name == 'fmri_cram_cpg_prop_attn_20':
        settings = Settings(
            corpora=(
                corpus_types.HarryPotterCorpus(
                    fmri_subjects=None,  # None means all
                    fmri_sentence_mode='ignore',
                    fmri_window_duration=10.1,
                    fmri_minimum_duration_required=9.6,
                    fmri_kind='rank_clustered',
                    fmri_smooth_factor=None,
                    separate_fmri_components=True,
                    group_meg_sentences_like_fmri=True,
                    meg_subjects=[]),
                corpus_types.BigramShift(),
                corpus_types.CoordinationInversion(),
                corpus_types.ObjectNumber(),
                corpus_types.SemanticOddManOut(),
                corpus_types.SentenceLength(),
                corpus_types.SubjectNumber(),
                corpus_types.TopConstituents(),
                corpus_types.TreeDepth(),
                corpus_types.VerbTense(),
                corpus_types.WordContent()),
            optimization_settings=OptimizationSettings(
                num_train_epochs=10,
                num_epochs_train_prediction_heads_only=0,
                num_final_epochs_train_prediction_heads_only=0,
                # learning_rate_head=1e-3,
                learning_rate=1e-4,
                learning_rate_schedule=learning_rate_schedules.LinearWarmupSqrtDecayLearningRateScheduleFactory(1000),
                train_batch_size=8,
                predict_batch_size=8,
                num_loader_workers=8),
            loss_tasks=set(),
            data_id_in_batch_keys=None,
            field_spec_replacers={corpus_types.HarryPotterCorpus.__name__: {'is_sequence': False}},
            weight_losses_by_inverse_example_counts=False,
            num_meta_learn_gradient_samples=10,
            num_meta_learn_no_gradient_samples=0,
            sampler_factory=BatchOneTaskProportionalSamplerFactory(500),
            num_runs=4)
        settings.common_graph_parts = OrderedDict(
            kv_layer=AttentionKeyValues(
                ('bert', 'sequence', 'all'),
                output_key_name='attn_key',
                output_value_name='attn_value',
                num_heads=20,
                num_key_channels=5,
                num_value_channels=1),
            context_attn=ContextAttention(
                'response_id',
                'num_response_data_fields',
                'attn_key',
                'attn_value',
                'sequence_all_bottleneck'),
            pooled_bottleneck=PooledFromSequence('sequence_all_bottleneck', 'pooled_all_bottleneck'))
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessQuantileDigitize(
                quantiles=2,
                stop_mode=None,
                metadata_example_group_by='fmri_runs',
                train_on_all=True,
                use_one_hot=False)]
        settings.critics[ResponseKind.hp_fmri] = critic_types.NamedTargetSingleBinaryCrossEntropyWithLogits()
        settings.default_pooled_source = 'pooled_all_bottleneck'
        settings.default_sequence_source = 'sequence_all_bottleneck'
        # settings.loss_tasks.add(ResponseKind.generic)
        # settings.loss_tasks.add(ResponseKind.hp_fmri)
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.generic)
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.hp_fmri)
        return settings
    elif name == 'fmri_cram_cpg_prop_attn_100':
        settings = Settings(
            corpora=(
                corpus_types.HarryPotterCorpus(
                    fmri_subjects=None,  # None means all
                    fmri_sentence_mode='ignore',
                    fmri_window_duration=10.1,
                    fmri_minimum_duration_required=9.6,
                    fmri_kind='rank_clustered',
                    fmri_smooth_factor=None,
                    separate_fmri_components=True,
                    group_meg_sentences_like_fmri=True,
                    meg_subjects=[]),
                corpus_types.BigramShift(),
                corpus_types.CoordinationInversion(),
                corpus_types.ObjectNumber(),
                corpus_types.SemanticOddManOut(),
                corpus_types.SentenceLength(),
                corpus_types.SubjectNumber(),
                corpus_types.TopConstituents(),
                corpus_types.TreeDepth(),
                corpus_types.VerbTense(),
                corpus_types.WordContent()),
            optimization_settings=OptimizationSettings(
                num_train_epochs=10,
                num_epochs_train_prediction_heads_only=0,
                num_final_epochs_train_prediction_heads_only=0,
                # learning_rate_head=1e-3,
                learning_rate=1e-4,
                learning_rate_schedule=learning_rate_schedules.LinearWarmupSqrtDecayLearningRateScheduleFactory(1000),
                train_batch_size=8,
                predict_batch_size=8,
                num_loader_workers=8),
            loss_tasks=set(),
            data_id_in_batch_keys=None,
            field_spec_replacers={corpus_types.HarryPotterCorpus.__name__: {'is_sequence': False}},
            weight_losses_by_inverse_example_counts=False,
            num_meta_learn_gradient_samples=10,
            num_meta_learn_no_gradient_samples=0,
            sampler_factory=BatchOneTaskProportionalSamplerFactory(500),
            num_runs=4)
        settings.common_graph_parts = OrderedDict(
            kv_layer=AttentionKeyValues(
                ('bert', 'sequence', 'all'),
                output_key_name='attn_key',
                output_value_name='attn_value',
                num_heads=100,
                num_key_channels=5,
                num_value_channels=1),
            context_attn=ContextAttention(
                'response_id',
                'num_response_data_fields',
                'attn_key',
                'attn_value',
                'sequence_all_bottleneck'),
            pooled_bottleneck=PooledFromSequence('sequence_all_bottleneck', 'pooled_all_bottleneck'))
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessQuantileDigitize(
                quantiles=2,
                stop_mode=None,
                metadata_example_group_by='fmri_runs',
                train_on_all=True,
                use_one_hot=False)]
        settings.critics[ResponseKind.hp_fmri] = critic_types.NamedTargetSingleBinaryCrossEntropyWithLogits()
        settings.default_pooled_source = 'pooled_all_bottleneck'
        settings.default_sequence_source = 'sequence_all_bottleneck'
        # settings.loss_tasks.add(ResponseKind.generic)
        # settings.loss_tasks.add(ResponseKind.hp_fmri)
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.generic)
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.hp_fmri)
        return settings
    elif name == 'fmri_cram_cpg_prop_20_softmax':
        settings = Settings(
            corpora=(
                corpus_types.HarryPotterCorpus(
                    fmri_subjects=None,  # None means all
                    fmri_sentence_mode='ignore',
                    fmri_window_duration=10.1,
                    fmri_minimum_duration_required=9.6,
                    fmri_kind='rank_clustered',
                    fmri_smooth_factor=None,
                    separate_fmri_components=True,
                    group_meg_sentences_like_fmri=True,
                    meg_subjects=[]),
                corpus_types.BigramShift(),
                corpus_types.CoordinationInversion(),
                corpus_types.ObjectNumber(),
                corpus_types.SemanticOddManOut(),
                corpus_types.SentenceLength(),
                corpus_types.SubjectNumber(),
                corpus_types.TopConstituents(),
                corpus_types.TreeDepth(),
                corpus_types.VerbTense(),
                corpus_types.WordContent()),
            optimization_settings=OptimizationSettings(
                num_train_epochs=20,
                num_epochs_train_prediction_heads_only=0,
                num_final_epochs_train_prediction_heads_only=0,
                # learning_rate_head=1e-3,
                learning_rate=1e-5,
                learning_rate_schedule=learning_rate_schedules.LinearWarmupSqrtDecayLearningRateScheduleFactory(2000),
                train_batch_size=8,
                predict_batch_size=8,
                num_loader_workers=8),
            loss_tasks=set(),
            data_id_in_batch_keys=None,
            field_spec_replacers={corpus_types.HarryPotterCorpus.__name__: {'is_sequence': False}},
            weight_losses_by_inverse_example_counts=False,
            num_meta_learn_gradient_samples=10,
            num_meta_learn_no_gradient_samples=0,
            sampler_factory=BatchOneTaskProportionalSamplerFactory(500),
            num_runs=4)
        settings.common_graph_parts = OrderedDict(
            multi_layer_bottleneck=MultiLayerBottleneck(
                ('bert', 'sequence', 'all'),
                output_name='multi_layer_sequence_bottleneck',
                num_bottleneck_channels=1,
                num_output_bottlenecks=10,
                num_hidden_bottlenecks=None,
                hidden_activation=None,
                should_norm_hidden=True,
                should_transpose_output=False,
                should_norm=False),
            contextual_bottleneck=ContextualBottleneckSum(
                'response_id',
                'num_response_data_fields',
                'multi_layer_sequence_bottleneck',
                'sequence_all_bottleneck',
                softmax_weights=True,
                softmax_temperature_schedule_fn=LinearDecreasingTemperatureSchedule(100, 500)),
            pooled_bottleneck=PooledFromSequence('sequence_all_bottleneck', 'pooled_all_bottleneck'))
        # remove any trend from individual runs before computing the median on the training data
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessQuantileDigitize(quantiles=2, stop_mode=None, use_one_hot=False)]
        settings.critics[ResponseKind.hp_fmri] = critic_types.NamedTargetSingleBinaryCrossEntropyWithLogits()
        settings.default_pooled_source = 'pooled_all_bottleneck'
        settings.default_sequence_source = 'sequence_all_bottleneck'
        # settings.loss_tasks.add(ResponseKind.generic)
        # settings.loss_tasks.add(ResponseKind.hp_fmri)
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.generic)
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.hp_fmri)
        return settings
    elif name == 'fmri_cram_cpg_prop_ling_basis':
        settings = Settings(
            corpora=(
                corpus_types.HarryPotterCorpus(
                    fmri_subjects=None,  # None means all
                    fmri_sentence_mode='ignore',
                    fmri_window_duration=10.1,
                    fmri_minimum_duration_required=9.6,
                    fmri_kind='rank_clustered',
                    fmri_smooth_factor=None,
                    separate_fmri_components=True,
                    group_meg_sentences_like_fmri=True,
                    meg_subjects=[]),
                corpus_types.BigramShift(),
                corpus_types.CoordinationInversion(),
                corpus_types.ObjectNumber(),
                corpus_types.SemanticOddManOut(),
                corpus_types.SentenceLength(),
                corpus_types.SubjectNumber(),
                corpus_types.TopConstituents(),
                corpus_types.TreeDepth(),
                corpus_types.VerbTense(),
                corpus_types.WordContent()),
            optimization_settings=OptimizationSettings(
                num_train_epochs=20,
                num_epochs_train_prediction_heads_only=0,
                num_final_epochs_train_prediction_heads_only=0,
                # learning_rate_head=1e-3,
                learning_rate=1e-5,
                learning_rate_schedule=learning_rate_schedules.LinearWarmupSqrtDecayLearningRateScheduleFactory(2000),
                train_batch_size=8,
                predict_batch_size=8,
                num_loader_workers=8),
            loss_tasks=set(),
            data_id_in_batch_keys=None,
            field_spec_replacers={corpus_types.HarryPotterCorpus.__name__: {'is_sequence': False}},
            weight_losses_by_inverse_example_counts=False,
            num_meta_learn_gradient_samples=10,
            num_meta_learn_no_gradient_samples=0,
            sampler_factory=BatchOneTaskProportionalSamplerFactory(500),
            num_runs=4)
        settings.head_graph_parts = OrderedDict()
        settings.head_graph_parts[ResponseKind.generic] = OrderedDict(
            ling_linear=KeyedLinear(
                ('bert', 'untransformed_pooled'), targets=(ResponseKind.generic,)),
            ling_concat=KeyedConcat([ResponseKind.generic], 'ling_basis'))
        settings.head_graph_parts[ResponseKind.hp_fmri] = OrderedDict(
            fmri_linear=KeyedLinear('ling_basis', targets=(ResponseKind.hp_fmri,)))
        # remove any trend from individual runs before computing the median on the training data
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessQuantileDigitize(quantiles=2, stop_mode=None, use_one_hot=False)]
        settings.critics[ResponseKind.hp_fmri] = critic_types.NamedTargetSingleBinaryCrossEntropyWithLogits()
        settings.default_pooled_source = 'pooled_all_bottleneck'
        settings.default_sequence_source = 'sequence_all_bottleneck'
        # settings.loss_tasks.add(ResponseKind.generic)
        # settings.loss_tasks.add(ResponseKind.hp_fmri)
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.generic)
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.hp_fmri)
        return settings
    elif name == 'fmri_only':
        settings = Settings(
            corpora=(
                corpus_types.HarryPotterCorpus(
                    fmri_subjects=None,  # None means all
                    fmri_sentence_mode='ignore',
                    fmri_window_duration=10.1,
                    fmri_minimum_duration_required=9.6,
                    fmri_kind='rank_clustered',
                    fmri_smooth_factor=None,
                    separate_fmri_components=True,
                    group_meg_sentences_like_fmri=True,
                    meg_subjects=[]),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=10,
                num_epochs_train_prediction_heads_only=0,
                num_final_epochs_train_prediction_heads_only=0,
                # learning_rate_head=1e-3,
                learning_rate=1e-5,
                learning_rate_schedule=learning_rate_schedules.LinearWarmupSqrtDecayLearningRateScheduleFactory(2000),
                train_batch_size=8,
                predict_batch_size=8,
                num_loader_workers=8),
            loss_tasks=set(),
            data_id_in_batch_keys=None,
            field_spec_replacers={corpus_types.HarryPotterCorpus.__name__: {'is_sequence': False}},
            weight_losses_by_inverse_example_counts=False,
            num_meta_learn_gradient_samples=10,
            num_meta_learn_no_gradient_samples=0,
            sampler_factory=BatchOneTaskProportionalSamplerFactory(500),
            num_runs=4)
        # settings.preprocessors[ResponseKind.hp_fmri] = [
        #     PreprocessQuantileDigitize(
        #         quantiles=2,
        #         stop_mode=None,
        #         metadata_example_group_by='fmri_runs',
        #         train_on_all=True,
        #         use_one_hot=False)]
        # settings.critics[ResponseKind.hp_fmri] = critic_types.NamedTargetSingleBinaryCrossEntropyWithLogits()
        settings.default_pooled_source = 'bert', 'untransformed_pooled'
        # settings.loss_tasks.add(ResponseKind.generic)
        # settings.loss_tasks.add(ResponseKind.hp_fmri)
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.hp_fmri)
        return settings
    elif name == 'fmri_only_common_median':
        settings = Settings(
            corpora=(
                corpus_types.HarryPotterCorpus(
                    fmri_subjects=None,  # None means all
                    fmri_sentence_mode='ignore',
                    fmri_window_duration=10.1,
                    fmri_minimum_duration_required=9.6,
                    fmri_kind='rank_clustered',
                    fmri_smooth_factor=None,
                    separate_fmri_components=True,
                    group_meg_sentences_like_fmri=True,
                    meg_subjects=[]),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=10,
                num_epochs_train_prediction_heads_only=0,
                num_final_epochs_train_prediction_heads_only=0,
                # learning_rate_head=1e-3,
                learning_rate=1e-5,
                learning_rate_schedule=learning_rate_schedules.LinearWarmupSqrtDecayLearningRateScheduleFactory(2000),
                train_batch_size=8,
                predict_batch_size=8,
                num_loader_workers=8),
            loss_tasks=set(),
            data_id_in_batch_keys=None,
            field_spec_replacers={corpus_types.HarryPotterCorpus.__name__: {'is_sequence': False}},
            weight_losses_by_inverse_example_counts=False,
            num_meta_learn_gradient_samples=10,
            num_meta_learn_no_gradient_samples=0,
            sampler_factory=BatchOneTaskProportionalSamplerFactory(500),
            num_runs=4)
        # remove any trend from individual runs before computing the median on the training data
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessQuantileDigitize(quantiles=2, stop_mode=None, use_one_hot=False)]
        settings.critics[ResponseKind.hp_fmri] = critic_types.NamedTargetSingleBinaryCrossEntropyWithLogits()
        settings.default_pooled_source = 'bert', 'untransformed_pooled'
        # settings.loss_tasks.add(ResponseKind.generic)
        # settings.loss_tasks.add(ResponseKind.hp_fmri)
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.hp_fmri)
        return settings
    elif name == 'fmri_cram_asymmetric':
        settings = Settings(
            corpora=(
                corpus_types.HarryPotterCorpus(
                    fmri_subjects=None,  # None means all
                    fmri_sentence_mode='ignore',
                    fmri_window_duration=10.1,
                    fmri_minimum_duration_required=9.6,
                    fmri_kind='rank_clustered',
                    fmri_smooth_factor=None,
                    separate_fmri_components=True,
                    group_meg_sentences_like_fmri=True,
                    meg_subjects=[]),
                corpus_types.BigramShift(),
                corpus_types.CoordinationInversion(),
                corpus_types.ObjectNumber(),
                corpus_types.SemanticOddManOut(),
                corpus_types.SentenceLength(),
                corpus_types.SubjectNumber(),
                corpus_types.TopConstituents(),
                corpus_types.TreeDepth(),
                corpus_types.VerbTense()),
                # corpus_types.WordContent()),
            optimization_settings=OptimizationSettings(
                num_train_epochs=20,
                num_epochs_train_prediction_heads_only=0,
                num_final_epochs_train_prediction_heads_only=0,
                # learning_rate_head=1e-3,
                learning_rate=1e-3,
                learning_rate_schedule=learning_rate_schedules.LinearWarmupSqrtDecayLearningRateScheduleFactory(2000),
                train_batch_size=8,
                predict_batch_size=8,
                num_loader_workers=8),
            loss_tasks=set(),
            data_id_in_batch_keys=None,
            field_spec_replacers={corpus_types.HarryPotterCorpus.__name__: {'is_sequence': False}},
            weight_losses_by_inverse_example_counts=False,
            num_meta_learn_gradient_samples=10,
            num_meta_learn_no_gradient_samples=0,
            sampler_factory=BatchOneTaskProportionalSamplerFactory(500),
            num_runs=4)
        settings.common_graph_parts = OrderedDict(
            attn_kv=AttentionKeyValues(
                source_name=('bert', 'sequence'),
                output_key_name='latent_keys',
                output_value_name='latent_values',
                num_heads=100,
                num_key_channels=10,
                num_value_channels=1,
                value_activation_fn=gelu),
            attn_pool=AttentionPool(
                'latent_keys', 'latent_values', 'bottleneck', should_layer_norm=False, flatten=True))
        settings.head_graph_parts = OrderedDict()
        settings.head_graph_parts[ResponseKind.generic] = OrderedDict(
            predictions=KeyedLinear(
                'bottleneck',
                targets=(ResponseKind.generic, ResponseKind.hp_fmri),
                hidden_activation=gelu,
                should_norm_hidden=False,
                penultimate_reconstruction_penalty_coefficient=1e-4,
                penultimate_reconstruction_l1_weight_coefficient=1e-4))
        settings.head_graph_parts[ResponseKind.hp_fmri] = settings.head_graph_parts[ResponseKind.generic]
        # remove any trend from individual runs before computing the median on the training data
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessQuantileDigitize(quantiles=2, stop_mode=None, use_one_hot=False)]
        settings.critics[ResponseKind.hp_fmri] = critic_types.NamedTargetSingleBinaryCrossEntropyWithLogits()
        settings.default_pooled_source = 'pooled_all_bottleneck'
        settings.default_sequence_source = 'sequence_all_bottleneck'
        # settings.loss_tasks.add(ResponseKind.generic)
        # settings.loss_tasks.add(ResponseKind.hp_fmri)
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.generic)
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.hp_fmri)
        return settings
    elif name == 'fmri_cram_ippr':
        settings = Settings(
            corpora=(
                corpus_types.HarryPotterCorpus(
                    fmri_subjects=None,  # None means all
                    fmri_sentence_mode='ignore',
                    fmri_window_duration=10.1,
                    fmri_minimum_duration_required=9.6,
                    fmri_kind='inter_participant_prediction_rank_clustered',
                    fmri_smooth_factor=None,
                    separate_fmri_components=True,
                    group_meg_sentences_like_fmri=True,
                    meg_subjects=[]),
                corpus_types.BigramShift(),
                corpus_types.CoordinationInversion(),
                corpus_types.ObjectNumber(),
                corpus_types.SemanticOddManOut(),
                corpus_types.SentenceLength(),
                corpus_types.SubjectNumber(),
                corpus_types.TopConstituents(),
                corpus_types.TreeDepth(),
                corpus_types.VerbTense()),
            # corpus_types.WordContent()),
            optimization_settings=OptimizationSettings(
                num_train_epochs=20,
                num_epochs_train_prediction_heads_only=0,
                num_final_epochs_train_prediction_heads_only=0,
                # learning_rate_head=1e-3,
                learning_rate=1e-3,
                learning_rate_schedule=learning_rate_schedules.LinearWarmupSqrtDecayLearningRateScheduleFactory(2000),
                train_batch_size=8,
                predict_batch_size=8,
                num_loader_workers=8),
            loss_tasks=set(),
            data_id_in_batch_keys=None,
            field_spec_replacers={corpus_types.HarryPotterCorpus.__name__: {'is_sequence': False}},
            weight_losses_by_inverse_example_counts=False,
            num_meta_learn_gradient_samples=10,
            num_meta_learn_no_gradient_samples=0,
            sampler_factory=BatchOneTaskProportionalSamplerFactory(500),
            num_runs=4)
        settings.common_graph_parts = OrderedDict(
            attn_kv=AttentionKeyValues(
                source_name=('bert', 'sequence'),
                output_key_name='latent_keys',
                output_value_name='latent_values',
                num_heads=100,
                num_key_channels=10,
                num_value_channels=1,
                value_activation_fn=gelu),
            attn_pool=AttentionPool(
                'latent_keys', 'latent_values', 'bottleneck', should_layer_norm=False, flatten=True))
        settings.head_graph_parts = OrderedDict()
        settings.head_graph_parts[ResponseKind.generic] = OrderedDict(
            predictions=KeyedLinear(
                'bottleneck',
                targets=(ResponseKind.generic, ResponseKind.hp_fmri),
                hidden_activation=gelu,
                should_norm_hidden=False,
                penultimate_reconstruction_penalty_coefficient=0,  # 1e-4,
                penultimate_reconstruction_l1_weight_coefficient=0))  # 1e-4))
        settings.head_graph_parts[ResponseKind.hp_fmri] = settings.head_graph_parts[ResponseKind.generic]
        # remove any trend from individual runs before computing the median on the training data
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessQuantileDigitize(quantiles=2, stop_mode=None, use_one_hot=False)]
        settings.critics[ResponseKind.hp_fmri] = critic_types.NamedTargetSingleBinaryCrossEntropyWithLogits()
        settings.default_pooled_source = 'pooled_all_bottleneck'
        settings.default_sequence_source = 'sequence_all_bottleneck'
        # settings.loss_tasks.add(ResponseKind.generic)
        # settings.loss_tasks.add(ResponseKind.hp_fmri)
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.generic)
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.hp_fmri)
        return settings
    elif name == 'fmri_cram_ippr_surrogate':
        settings = Settings(
            corpora=(
                corpus_types.HarryPotterCorpus(
                    fmri_subjects=None,  # None means all
                    fmri_sentence_mode='ignore',
                    fmri_window_duration=10.1,
                    fmri_minimum_duration_required=9.6,
                    fmri_kind='inter_participant_prediction_surrogate_rank_clustered',
                    fmri_smooth_factor=None,
                    separate_fmri_components=True,
                    group_meg_sentences_like_fmri=True,
                    meg_subjects=[]),
                corpus_types.BigramShift(),
                corpus_types.CoordinationInversion(),
                corpus_types.ObjectNumber(),
                corpus_types.SemanticOddManOut(),
                corpus_types.SentenceLength(),
                corpus_types.SubjectNumber(),
                corpus_types.TopConstituents(),
                corpus_types.TreeDepth(),
                corpus_types.VerbTense()),
            # corpus_types.WordContent()),
            optimization_settings=OptimizationSettings(
                num_train_epochs=20,
                num_epochs_train_prediction_heads_only=0,
                num_final_epochs_train_prediction_heads_only=0,
                # learning_rate_head=1e-3,
                learning_rate=1e-3,
                learning_rate_schedule=learning_rate_schedules.LinearWarmupSqrtDecayLearningRateScheduleFactory(2000),
                train_batch_size=8,
                predict_batch_size=8,
                num_loader_workers=8),
            loss_tasks=set(),
            data_id_in_batch_keys=None,
            field_spec_replacers={corpus_types.HarryPotterCorpus.__name__: {'is_sequence': False}},
            weight_losses_by_inverse_example_counts=False,
            num_meta_learn_gradient_samples=10,
            num_meta_learn_no_gradient_samples=0,
            sampler_factory=BatchOneTaskProportionalSamplerFactory(500),
            num_runs=4)
        settings.common_graph_parts = OrderedDict(
            attn_kv=AttentionKeyValues(
                source_name=('bert', 'sequence'),
                output_key_name='latent_keys',
                output_value_name='latent_values',
                num_heads=100,
                num_key_channels=10,
                num_value_channels=1,
                value_activation_fn=gelu),
            attn_pool=AttentionPool(
                'latent_keys', 'latent_values', 'bottleneck', should_layer_norm=False, flatten=True))
        settings.head_graph_parts = OrderedDict()
        settings.head_graph_parts[ResponseKind.generic] = OrderedDict(
            predictions=KeyedLinear(
                'bottleneck',
                targets=(ResponseKind.generic, ResponseKind.hp_fmri),
                hidden_activation=gelu,
                should_norm_hidden=False,
                penultimate_reconstruction_penalty_coefficient=0,  # 1e-4,
                penultimate_reconstruction_l1_weight_coefficient=0))  # 1e-4))
        settings.head_graph_parts[ResponseKind.hp_fmri] = settings.head_graph_parts[ResponseKind.generic]
        # remove any trend from individual runs before computing the median on the training data
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessQuantileDigitize(quantiles=2, stop_mode=None, use_one_hot=False)]
        settings.critics[ResponseKind.hp_fmri] = critic_types.NamedTargetSingleBinaryCrossEntropyWithLogits()
        settings.default_pooled_source = 'pooled_all_bottleneck'
        settings.default_sequence_source = 'sequence_all_bottleneck'
        # settings.loss_tasks.add(ResponseKind.generic)
        # settings.loss_tasks.add(ResponseKind.hp_fmri)
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.generic)
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.hp_fmri)
        return settings
    elif name == 'fmri_cram_cpg_ippr':
        settings = Settings(
            corpora=(
                corpus_types.HarryPotterCorpus(
                    fmri_subjects=None,  # None means all
                    fmri_sentence_mode='ignore',
                    fmri_window_duration=10.1,
                    fmri_minimum_duration_required=9.6,
                    fmri_kind='inter_participant_prediction_rank_clustered',
                    fmri_smooth_factor=None,
                    separate_fmri_components=True,
                    group_meg_sentences_like_fmri=True,
                    meg_subjects=[]),
                corpus_types.BigramShift(),
                corpus_types.CoordinationInversion(),
                corpus_types.ObjectNumber(),
                corpus_types.SemanticOddManOut(),
                corpus_types.SentenceLength(),
                corpus_types.SubjectNumber(),
                corpus_types.TopConstituents(),
                corpus_types.TreeDepth(),
                corpus_types.VerbTense()),
                # corpus_types.WordContent()),
            optimization_settings=OptimizationSettings(
                num_train_epochs=10,
                num_epochs_train_prediction_heads_only=0,
                num_final_epochs_train_prediction_heads_only=0,
                learning_rate_head=1e-3,
                learning_rate=1e-5,
                learning_rate_schedule=learning_rate_schedules.LinearWarmupSqrtDecayLearningRateScheduleFactory(1000),
                train_batch_size=8,
                predict_batch_size=8,
                num_loader_workers=8),
            loss_tasks=set(),
            data_id_in_batch_keys=None,
            field_spec_replacers={corpus_types.HarryPotterCorpus.__name__: {'is_sequence': False}},
            weight_losses_by_inverse_example_counts=False,
            num_meta_learn_gradient_samples=10,
            num_meta_learn_no_gradient_samples=0,
            sampler_factory=BatchOneTaskProportionalSamplerFactory(500),
            num_runs=4)
        settings.common_graph_parts = OrderedDict(
            contextual_bottleneck=LinearContextualParameterGeneration(
                'response_id', 'num_response_data_fields', 20,
                OrderedDict(
                    bottleneck=KeyedLinear(
                        ('bert', 'sequence', 'all'),
                        output_key_to_shape=OrderedDict(sequence_all_bottleneck=10),
                        should_norm=True))),
            pooled_bottleneck=PooledFromSequence('sequence_all_bottleneck', 'pooled_all_bottleneck'))
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessQuantileDigitize(
                quantiles=2,
                stop_mode=None,
                metadata_example_group_by='fmri_runs',
                train_on_all=True,
                use_one_hot=False)]
        settings.critics[ResponseKind.hp_fmri] = critic_types.NamedTargetSingleBinaryCrossEntropyWithLogits()
        settings.default_pooled_source = 'pooled_all_bottleneck'
        settings.default_sequence_source = 'sequence_all_bottleneck'
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.generic)
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.hp_fmri)
        return settings
    elif name == 'fmri_cram_cpg_ippr_uniform':
        settings = Settings(
            corpora=(
                corpus_types.HarryPotterCorpus(
                    fmri_subjects=None,  # None means all
                    fmri_sentence_mode='ignore',
                    fmri_window_duration=10.1,
                    fmri_minimum_duration_required=9.6,
                    fmri_kind='inter_participant_prediction_rank_clustered',
                    fmri_smooth_factor=None,
                    separate_fmri_components=True,
                    group_meg_sentences_like_fmri=True,
                    meg_subjects=[]),
                corpus_types.BigramShift(),
                corpus_types.CoordinationInversion(),
                corpus_types.ObjectNumber(),
                corpus_types.SemanticOddManOut(),
                corpus_types.SentenceLength(),
                corpus_types.SubjectNumber(),
                corpus_types.TopConstituents(),
                corpus_types.TreeDepth(),
                corpus_types.VerbTense()),
            # corpus_types.WordContent()),
            optimization_settings=OptimizationSettings(
                num_train_epochs=10,
                num_epochs_train_prediction_heads_only=0,
                num_final_epochs_train_prediction_heads_only=0,
                learning_rate_head=1e-3,
                learning_rate=1e-5,
                learning_rate_schedule=learning_rate_schedules.LinearWarmupSqrtDecayLearningRateScheduleFactory(1000),
                train_batch_size=8,
                predict_batch_size=8,
                num_loader_workers=8),
            loss_tasks=set(),
            data_id_in_batch_keys=None,
            field_spec_replacers={corpus_types.HarryPotterCorpus.__name__: {'is_sequence': False}},
            weight_losses_by_inverse_example_counts=False,
            num_meta_learn_gradient_samples=10,
            num_meta_learn_no_gradient_samples=0,
            sampler_factory=BatchOneTaskSamplerFactory(500),
            num_runs=4)
        settings.common_graph_parts = OrderedDict(
            contextual_bottleneck=LinearContextualParameterGeneration(
                'response_id', 'num_response_data_fields', 20,
                OrderedDict(
                    bottleneck=KeyedLinear(
                        ('bert', 'sequence', 'all'),
                        output_key_to_shape=OrderedDict(sequence_all_bottleneck=10),
                        should_norm=True)),
                use_softmax_embedding=True),
            pooled_bottleneck=PooledFromSequence('sequence_all_bottleneck', 'pooled_all_bottleneck'))
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessQuantileDigitize(
                quantiles=2,
                stop_mode=None,
                metadata_example_group_by='fmri_runs',
                train_on_all=True,
                use_one_hot=False)]
        settings.critics[ResponseKind.hp_fmri] = critic_types.NamedTargetSingleBinaryCrossEntropyWithLogits()
        settings.default_pooled_source = 'pooled_all_bottleneck'
        settings.default_sequence_source = 'sequence_all_bottleneck'
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.generic)
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.hp_fmri)
        return settings
    elif name == 'fmri_cram_cpg_maxima':
        settings = Settings(
            corpora=(
                corpus_types.HarryPotterCorpus(
                    fmri_subjects=None,  # None means all
                    fmri_sentence_mode='ignore',
                    fmri_window_duration=10.1,
                    fmri_minimum_duration_required=9.6,
                    fmri_kind='local_maxima_projection_clustered',
                    fmri_smooth_factor=None,
                    separate_fmri_components=True,
                    group_meg_sentences_like_fmri=True,
                    meg_subjects=[]),
                corpus_types.UclCorpus(subtract_erp_baseline=True),
                corpus_types.GhentEyeTrackingCorpus(active_fields=(
                    'word_fixation_count',
                    'word_first_fixation_duration',
                    'word_gaze_duration',
                    'word_go_past_time',
                    'word_total_reading_time')),
                corpus_types.DundeeCorpus(),
                corpus_types.StanfordSentimentTreebank(),
                corpus_types.BigramShift(),
                corpus_types.CoordinationInversion(),
                corpus_types.ObjectNumber(),
                corpus_types.SemanticOddManOut(),
                corpus_types.SentenceLength(),
                corpus_types.SubjectNumber(),
                corpus_types.TopConstituents(),
                corpus_types.TreeDepth(),
                corpus_types.VerbTense()),
            # corpus_types.WordContent()),
            optimization_settings=OptimizationSettings(
                num_train_epochs=10,
                num_epochs_train_prediction_heads_only=0,
                num_final_epochs_train_prediction_heads_only=0,
                learning_rate_head=1e-3,
                learning_rate=1e-5,
                learning_rate_schedule=learning_rate_schedules.LinearWarmupSqrtDecayLearningRateScheduleFactory(10000),
                train_batch_size=8,
                predict_batch_size=8,
                num_loader_workers=8),
            loss_tasks=set(),
            data_id_in_batch_keys=None,
            field_spec_replacers={corpus_types.HarryPotterCorpus.__name__: {'is_sequence': False}},
            weight_losses_by_inverse_example_counts=False,
            # num_meta_learn_gradient_samples=10,
            # num_meta_learn_no_gradient_samples=0,
            sampler_factory=BatchOneTaskSamplerFactory(5000),
            num_runs=4)
        settings.common_graph_parts = OrderedDict(
            contextual_bottleneck=LinearContextualParameterGeneration(
                'response_id', 'num_response_data_fields', 10,
                OrderedDict(
                    bottleneck=KeyedLinear(
                        ('bert', 'sequence', 'all'),
                        output_key_to_shape=OrderedDict(sequence_all_bottleneck=1),
                        should_norm=False)),
                use_softmax_embedding=True),
            pooled_bottleneck=PooledFromSequence('sequence_all_bottleneck', 'pooled_all_bottleneck'))
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessQuantileDigitize(
                quantiles=2,
                stop_mode=None,
                metadata_example_group_by='fmri_runs',
                train_on_all=True,
                use_one_hot=False)]
        settings.critics[ResponseKind.hp_fmri] = critic_types.NamedTargetSingleBinaryCrossEntropyWithLogits()
        for kind in [
                ResponseKind.ucl_erp,
                ResponseKind.ucl_eye,
                ResponseKind.ucl_self_paced,
                ResponseKind.dundee_eye,
                ResponseKind.geco]:
            settings.preprocessors[kind] = [
                # use standardize to average subjects
                PreprocessStandardize(stop_mode='content'),
                PreprocessQuantileDigitize(
                    quantiles=2,
                    stop_mode='content',
                    use_one_hot=False)]
            settings.critics[kind] = critic_types.NamedTargetStopWordAwareBinaryCrossEntropyWithLogits()
        settings.default_pooled_source = 'pooled_all_bottleneck'
        settings.default_sequence_source = 'sequence_all_bottleneck'
        settings.loss_tasks.add(ResponseKind.generic)
        settings.loss_tasks.add(ResponseKind.hp_fmri)
        settings.loss_tasks.add(ResponseKind.ucl_erp)
        settings.loss_tasks.add(ResponseKind.ucl_eye)
        settings.loss_tasks.add(ResponseKind.ucl_self_paced)
        settings.loss_tasks.add(ResponseKind.dundee_eye)
        settings.loss_tasks.add(ResponseKind.geco)

        # settings.meta_learn_gradient_loss_tasks.add(ResponseKind.generic)
        # settings.meta_learn_gradient_loss_tasks.add(ResponseKind.hp_fmri)
        return settings
    elif name == 'fmri_cram_cpg_cca_ica':
        settings = Settings(
            corpora=(
                corpus_types.HarryPotterCorpus(
                    fmri_subjects=None,  # None means all
                    fmri_sentence_mode='ignore',
                    fmri_window_duration=10.1,
                    fmri_minimum_duration_required=9.6,
                    fmri_kind='cca_ica',
                    fmri_smooth_factor=None,
                    separate_fmri_components=True,
                    group_meg_sentences_like_fmri=True,
                    meg_subjects=[]),
                corpus_types.UclCorpus(subtract_erp_baseline=True),
                corpus_types.GhentEyeTrackingCorpus(active_fields=(
                    'word_fixation_count',
                    'word_first_fixation_duration',
                    'word_gaze_duration',
                    'word_go_past_time',
                    'word_total_reading_time')),
                corpus_types.DundeeCorpus(),
                corpus_types.StanfordSentimentTreebank(),
                corpus_types.BigramShift(),
                corpus_types.CoordinationInversion(),
                corpus_types.ObjectNumber(),
                corpus_types.SemanticOddManOut(),
                corpus_types.SentenceLength(),
                corpus_types.SubjectNumber(),
                corpus_types.TopConstituents(),
                corpus_types.TreeDepth(),
                corpus_types.VerbTense(),
                corpus_types.PartOfSpeechConll2012(),
                corpus_types.ConstituentsConll2012(),
                corpus_types.SemanticRoleLabelConll2012(),
                corpus_types.NamedEntityRecognitionConll2012(),
                corpus_types.CoreferenceResolutionConll2012(),
                corpus_types.DependenciesEnglishWeb(),
                corpus_types.DefinitePronounResolution(),
                corpus_types.SemEval(),
                corpus_types.SemanticProtoRoles1(),
                corpus_types.SemanticProtoRoles2()),
            # corpus_types.WordContent()),
            optimization_settings=OptimizationSettings(
                num_train_epochs=40,
                num_epochs_train_prediction_heads_only=0,
                num_final_epochs_train_prediction_heads_only=0,
                learning_rate_head=1e-4,
                learning_rate=1e-5,
                learning_rate_schedule=learning_rate_schedules.LinearWarmupSqrtDecayLearningRateScheduleFactory(2000),
                train_batch_size=8,
                predict_batch_size=8,
                num_loader_workers=8),
            loss_tasks=set(),
            data_id_in_batch_keys=None,
            field_spec_replacers={corpus_types.HarryPotterCorpus.__name__: {'is_sequence': False}},
            weight_losses_by_inverse_example_counts=False,
            sampler_factory=BatchOneTaskManualWeightSamplerFactory(
                1000, weight_fn=SamplerWeightCollapse(('spr1.', 'spr2.'), divisor=100, temperature=5)),
            # sampler_factory=BatchOneTaskTaskPermutedSamplerFactory(10),
            # sampler_factory=BatchOneTaskTemperatureProportionalSamplerFactory(1000, temperature=5),
            num_runs=4)
        settings.common_graph_parts = OrderedDict(
            contextual_bottleneck=LinearContextualParameterGeneration(
                'response_id', 'num_response_data_fields', 5,
                OrderedDict(
                    bottleneck=KeyedLinear(
                        ('bert', 'sequence', 'all'),
                        output_key_to_shape=OrderedDict(sequence_all_bottleneck=3),
                        should_norm=False)),
                use_softmax_embedding=True),
            pooled_bottleneck=PooledFromSequence('sequence_all_bottleneck', 'pooled_all_bottleneck'),
            hdr_bottleneck=PooledFromKTokens(
                num_tokens=20, source_name='sequence_all_bottleneck', output_name='hdr_pooled'))

        settings.head_graph_parts.update(standard_edge_probing_graph(settings.corpora, 'sequence_all_bottleneck'))

        settings.head_graph_parts[ResponseKind.hp_fmri] = OrderedDict(
            fmri_linear=KeyedLinear(
                'hdr_pooled',
                apply_at_most_one_data_id='if_no_target',
                targets=ResponseKind.hp_fmri))
        # settings.preprocessors[ResponseKind.hp_fmri] = [
        #     # PreprocessQuantileDigitize(quantiles=10, use_one_hot=False),
        #     PreprocessStandardize(metadata_example_group_by='fmri_runs', train_on_all=True, average_axis=None)]
        # settings.critics[ResponseKind.hp_fmri] = critic_types.NamedTargetSingleMSE()
        # for kind in [
        #         ResponseKind.ucl_eye,
        #         ResponseKind.ucl_self_paced,
        #         ResponseKind.dundee_eye,
        #         ResponseKind.geco]:
        #     # settings.preprocessors[kind] = [
        #     #     PreprocessLog(),
        #     #     PreprocessStandardize(stop_mode='content')]
        #     settings.preprocessors[kind] = [
        #         PreprocessLog(),
        #         PreprocessStandardize(stop_mode='content')]
        #         # PreprocessQuantileDigitize(quantiles=10, use_one_hot=False, stop_mode='content'),
        #         # PreprocessStandardize(stop_mode='content', average_axis=None)]
        # # settings.preprocessors[ResponseKind.ucl_erp] = [PreprocessStandardize(stop_mode='content')]
        # settings.preprocessors[ResponseKind.ucl_erp] = [
        #     PreprocessStandardize(stop_mode='content'),]  # use standardize to average
        #     # PreprocessQuantileDigitize(quantiles=19, use_one_hot=False, stop_mode='content'),
        #     # PreprocessStandardize(stop_mode='content', average_axis=None)]
        # for kind in [
        #         ResponseKind.ucl_erp,
        #         ResponseKind.ucl_eye,
        #         ResponseKind.ucl_self_paced,
        #         ResponseKind.dundee_eye,
        #         ResponseKind.geco]:
        #     settings.critics[kind] = critic_types.NamedTargetStopWordAwareMSE()
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessQuantileDigitize(
                quantiles=2,
                stop_mode=None,
                metadata_example_group_by='fmri_runs',
                train_on_all=True,
                use_one_hot=False)]
        settings.critics[ResponseKind.hp_fmri] = critic_types.NamedTargetSingleBinaryCrossEntropyWithLogits()
        for kind in [
                ResponseKind.ucl_erp,
                ResponseKind.ucl_eye,
                ResponseKind.ucl_self_paced,
                ResponseKind.dundee_eye,
                ResponseKind.geco]:
            settings.preprocessors[kind] = [
                # use standardize to average subjects
                PreprocessStandardize(stop_mode='content'),
                PreprocessQuantileDigitize(
                    quantiles=2,
                    stop_mode='content',
                    use_one_hot=False)]
            settings.critics[kind] = critic_types.NamedTargetStopWordAwareBinaryCrossEntropyWithLogits()
        settings.default_pooled_source = 'pooled_all_bottleneck'
        settings.default_sequence_source = 'sequence_all_bottleneck'
        settings.loss_tasks.add(ResponseKind.generic)
        settings.loss_tasks.add(ResponseKind.hp_fmri)
        settings.loss_tasks.add(ResponseKind.ucl_erp)
        settings.loss_tasks.add(ResponseKind.ucl_eye)
        settings.loss_tasks.add(ResponseKind.ucl_self_paced)
        settings.loss_tasks.add(ResponseKind.dundee_eye)
        settings.loss_tasks.add(ResponseKind.geco)

        return settings
    elif name == 'fmri_cram_cpg_cca_ica_meta':
        settings = Settings(
            corpora=(
                corpus_types.HarryPotterCorpus(
                    fmri_subjects=None,  # None means all
                    fmri_sentence_mode='ignore',
                    fmri_window_duration=10.1,
                    fmri_minimum_duration_required=9.6,
                    fmri_kind='cca_ica',
                    fmri_smooth_factor=None,
                    separate_fmri_components=True,
                    group_meg_sentences_like_fmri=True,
                    meg_subjects=[]),
                corpus_types.UclCorpus(subtract_erp_baseline=True),
                corpus_types.GhentEyeTrackingCorpus(active_fields=(
                    'word_fixation_count',
                    'word_first_fixation_duration',
                    'word_gaze_duration',
                    'word_go_past_time',
                    'word_total_reading_time')),
                corpus_types.DundeeCorpus(),
                corpus_types.StanfordSentimentTreebank(),
                corpus_types.BigramShift(),
                corpus_types.CoordinationInversion(),
                corpus_types.ObjectNumber(),
                corpus_types.SemanticOddManOut(),
                corpus_types.SentenceLength(),
                corpus_types.SubjectNumber(),
                corpus_types.TopConstituents(),
                corpus_types.TreeDepth(),
                corpus_types.VerbTense(),
                corpus_types.PartOfSpeechConll2012(),
                corpus_types.ConstituentsConll2012(),
                corpus_types.SemanticRoleLabelConll2012(),
                corpus_types.NamedEntityRecognitionConll2012(),
                corpus_types.CoreferenceResolutionConll2012(),
                corpus_types.DependenciesEnglishWeb(),
                corpus_types.DefinitePronounResolution(),
                corpus_types.SemEval(),
                corpus_types.SemanticProtoRoles1(),
                corpus_types.SemanticProtoRoles2()),
            # corpus_types.WordContent()),
            optimization_settings=OptimizationSettings(
                num_train_epochs=40,
                num_epochs_train_prediction_heads_only=0,
                num_final_epochs_train_prediction_heads_only=0,
                learning_rate_head=1e-4,
                learning_rate=1e-5,
                learning_rate_schedule=learning_rate_schedules.LinearWarmupSqrtDecayLearningRateScheduleFactory(200),
                train_batch_size=8,
                predict_batch_size=8,
                num_loader_workers=8),
            loss_tasks=set(),
            data_id_in_batch_keys=None,
            field_spec_replacers={corpus_types.HarryPotterCorpus.__name__: {'is_sequence': False}},
            weight_losses_by_inverse_example_counts=False,
            sampler_factory=BatchOneTaskManualWeightSamplerFactory(
                100, weight_fn=SamplerWeightCollapse(('spr1.', 'spr2.'), divisor=100, temperature=5)),
            # sampler_factory=BatchOneTaskTaskPermutedSamplerFactory(10),
            # sampler_factory=BatchOneTaskTemperatureProportionalSamplerFactory(1000, temperature=5),
            num_runs=4)
        settings.common_graph_parts = OrderedDict(
            contextual_bottleneck=LinearContextualParameterGeneration(
                'response_id', 'num_response_data_fields', 5,
                OrderedDict(
                    bottleneck=KeyedLinear(
                        ('bert', 'sequence', 'all'),
                        output_key_to_shape=OrderedDict(sequence_all_bottleneck=3),
                        should_norm=False)),
                use_softmax_embedding=True),
            pooled_bottleneck=PooledFromSequence('sequence_all_bottleneck', 'pooled_all_bottleneck'),
            hdr_bottleneck=PooledFromKTokens(
                num_tokens=20, source_name='sequence_all_bottleneck', output_name='hdr_pooled'))

        settings.head_graph_parts.update(standard_edge_probing_graph(settings.corpora, 'sequence_all_bottleneck'))

        settings.head_graph_parts[ResponseKind.hp_fmri] = OrderedDict(
            fmri_linear=KeyedLinear(
                'hdr_pooled',
                apply_at_most_one_data_id='if_no_target',
                targets=ResponseKind.hp_fmri))
        # settings.preprocessors[ResponseKind.hp_fmri] = [
        #     # PreprocessQuantileDigitize(quantiles=10, use_one_hot=False),
        #     PreprocessStandardize(metadata_example_group_by='fmri_runs', train_on_all=True, average_axis=None)]
        # settings.critics[ResponseKind.hp_fmri] = critic_types.NamedTargetSingleMSE()
        # for kind in [
        #         ResponseKind.ucl_eye,
        #         ResponseKind.ucl_self_paced,
        #         ResponseKind.dundee_eye,
        #         ResponseKind.geco]:
        #     # settings.preprocessors[kind] = [
        #     #     PreprocessLog(),
        #     #     PreprocessStandardize(stop_mode='content')]
        #     settings.preprocessors[kind] = [
        #         PreprocessLog(),
        #         PreprocessStandardize(stop_mode='content')]
        #         # PreprocessQuantileDigitize(quantiles=10, use_one_hot=False, stop_mode='content'),
        #         # PreprocessStandardize(stop_mode='content', average_axis=None)]
        # # settings.preprocessors[ResponseKind.ucl_erp] = [PreprocessStandardize(stop_mode='content')]
        # settings.preprocessors[ResponseKind.ucl_erp] = [
        #     PreprocessStandardize(stop_mode='content'),]  # use standardize to average
        #     # PreprocessQuantileDigitize(quantiles=19, use_one_hot=False, stop_mode='content'),
        #     # PreprocessStandardize(stop_mode='content', average_axis=None)]
        # for kind in [
        #         ResponseKind.ucl_erp,
        #         ResponseKind.ucl_eye,
        #         ResponseKind.ucl_self_paced,
        #         ResponseKind.dundee_eye,
        #         ResponseKind.geco]:
        #     settings.critics[kind] = critic_types.NamedTargetStopWordAwareMSE()
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessQuantileDigitize(
                quantiles=2,
                stop_mode=None,
                metadata_example_group_by='fmri_runs',
                train_on_all=True,
                use_one_hot=False)]
        settings.critics[ResponseKind.hp_fmri] = critic_types.NamedTargetSingleBinaryCrossEntropyWithLogits()
        for kind in [
                ResponseKind.ucl_erp,
                ResponseKind.ucl_eye,
                ResponseKind.ucl_self_paced,
                ResponseKind.dundee_eye,
                ResponseKind.geco]:
            settings.preprocessors[kind] = [
                # use standardize to average subjects
                PreprocessStandardize(stop_mode='content'),
                PreprocessQuantileDigitize(
                    quantiles=2,
                    stop_mode='content',
                    use_one_hot=False)]
            settings.critics[kind] = critic_types.NamedTargetStopWordAwareBinaryCrossEntropyWithLogits()
        settings.default_pooled_source = 'pooled_all_bottleneck'
        settings.default_sequence_source = 'sequence_all_bottleneck'

        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.generic)
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.hp_fmri)
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.ucl_erp)
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.ucl_eye)
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.ucl_self_paced)
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.dundee_eye)
        settings.meta_learn_gradient_loss_tasks.add(ResponseKind.geco)
        settings.num_meta_learn_gradient_samples = 10
        settings.num_meta_learn_no_gradient_samples = 0
        settings.use_pareto = False

        return settings
    else:
        raise ValueError('Unknown name: {}. Valid choices are: \n{}'.format(name.var, '\n'.join(name.tests)))
