import hashlib
import itertools
import random
from collections import OrderedDict
from dataclasses import replace
from typing import Union, Iterable, Mapping, Tuple

import numpy as np
import torch
import torch.cuda

from .common import SwitchRemember
from .data_sets import ResponseKind, PreprocessDetrend, PreprocessStandardize, \
    PreprocessKMeans, PreprocessRandomPair, PreprocessMakeBinary, PreprocessForkNoClusterToDisk, \
    PreprocessQuantileDigitize, corpus_types
from .modeling import KeyedLinear, CriticKeys, LinearContextualParameterGeneration, PooledFromSequence, \
    MarkedTokenConcatFixedNumTokens, GroupMultipart, KeyedSingleTargetSpanAttention
from .settings import Settings, OptimizationSettings, CriticSettings, LearningRateSchedule


__all__ = ['task_hash', 'set_random_seeds', 'iterate_powerset', 'named_variations', 'singleton_variation']


def superglue_heads(corpus, sequence_key=None, pooled_key=None):

    if pooled_key is None:
        pooled_key = ('bert', 'pooled')
    if sequence_key is None:
        sequence_key = ('bert', 'sequence')

    response_key = type(corpus).response_key()

    head = OrderedDict()
    critic = OrderedDict()
    if isinstance(corpus, corpus_types.CommitmentBank):
        head['{}_linear'.format(response_key)] = KeyedLinear(
            pooled_key, is_sequence=False, output_key_to_shape={response_key: 1}, apply_at_most_one_data_id=True)
        critic[response_key] = CriticSettings(
            critic_type=CriticKeys.single_cross_entropy, critic_kwargs=dict(num_classes=4))
        return head, critic
    if isinstance(corpus, (corpus_types.BooleanQuestions, corpus_types.RecognizingTextualEntailment)):
        head['{}_linear'.format(response_key)] = KeyedLinear(
            pooled_key, is_sequence=False, output_key_to_shape={response_key: 1}, apply_at_most_one_data_id=True)
        critic[response_key] = CriticSettings(critic_type=CriticKeys.single_binary_cross_entropy)
        return head, critic
    elif isinstance(corpus, corpus_types.WordInContext):
        head['{}_group'.format(response_key)] = MarkedTokenConcatFixedNumTokens(
            2,
            response_key, 'data_ids',
            '{}_concat'.format(response_key),
            sequence_key)
        head['{}_linear'.format(response_key)] = KeyedLinear(
            '{}_concat'.format(response_key), is_sequence=False,
            output_key_to_shape={response_key: 1}, apply_at_most_one_data_id=True)
        critic[response_key] = CriticSettings(critic_type=CriticKeys.single_binary_cross_entropy)
        return head, critic
    elif isinstance(
            corpus,
            (corpus_types.ChoiceOfPlausibleAlternatives,
             corpus_types.ReadingComprehensionWithCommonSenseReasoning,
             corpus_types.MultiSentenceReadingComprehension)):
        head['{}_linear'.format(response_key)] = KeyedLinear(
            pooled_key, is_sequence=False, output_key_to_shape={
                '{}_choice'.format(response_key): 1}, apply_at_most_one_data_id=True)
        head['{}_mc'.format(response_key)] = GroupMultipart(
            None, 'multipart_id', response_key, '{}_choice'.format(response_key))
        critic[response_key] = CriticSettings(critic_type=CriticKeys.single_binary_cross_entropy)
        return head, critic
    elif isinstance(corpus, corpus_types.WinogradSchemaChallenge):
        head['{}_span_linear'.format(response_key)] = KeyedSingleTargetSpanAttention(
            2, sequence_key, 'span_ids', conv_hidden_channels=1024, conv_hidden_kernel=1,
            output_key_to_shape={response_key: 1})
        critic[response_key] = CriticSettings(critic_type=CriticKeys.single_binary_cross_entropy)
        return head, critic
    else:
        raise ValueError('Unknown corpus type: {}'.format(type(corpus)))


def what_you_can_cram_heads(corpus, pooled_key=None):

    if pooled_key is None:
        pooled_key = ('bert', 'pooled')

    head = OrderedDict()
    critic = OrderedDict()

    if isinstance(
            corpus,
            (corpus_types.BigramShift,
             corpus_types.CoordinationInversion,
             corpus_types.ObjectNumber,
             corpus_types.SemanticOddManOut,
             corpus_types.SentenceLength,
             corpus_types.SubjectNumber,
             corpus_types.TopConstituents,
             corpus_types.TreeDepth,
             corpus_types.VerbTense,
             corpus_types.WordContent)):
        response_key = type(corpus).response_key()
        num_classes = type(corpus).num_classes()
        head['{}_linear'.format(response_key)] = KeyedLinear(
            pooled_key, is_sequence=False, output_key_to_shape={response_key: 1}, apply_at_most_one_data_id=True)
        if num_classes > 2:
            critic[response_key] = CriticSettings(
                critic_type=CriticKeys.single_cross_entropy, critic_kwargs=dict(num_classes=num_classes))
        else:
            critic[response_key] = CriticSettings(critic_type=CriticKeys.single_binary_cross_entropy)
        return head, critic


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

    erp_tasks = ('epnp', 'pnp', 'elan', 'lan', 'n400', 'p600')
    ns_froi_tasks = (
        'ns_lh_pt', 'ns_lh_at', 'ns_lh_ifg', 'ns_lh_ifgpo', 'ns_lh_mfg', 'ns_lh_ag',
        'ns_rh_pt', 'ns_rh_at', 'ns_rh_ifg', 'ns_rh_ifgpo', 'ns_rh_mfg', 'ns_rh_ag')

    hp_fmri_subjects = ('F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N')
    hp_meg_subjects = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I')
    hp_fmri_tasks = tuple('hp_fmri_{}'.format(s) for s in hp_fmri_subjects)

    name = SwitchRemember(name)

    if name == 'erp':
        return [
            Settings(corpora=(corpus_types.UclCorpus(),), num_runs=100, loss_tasks=set(t))
            for t in iterate_powerset(erp_tasks)]
    elif name == 'erp_joint':
        return Settings(corpora=(corpus_types.UclCorpus(),), loss_tasks=set(erp_tasks), num_runs=100)
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
                ('bert', 'untransformed_pooled'), is_sequence=False, apply_at_most_one_data_id='if_no_target',
                targets=ResponseKind.ns_froi))
        return [replace(settings, loss_tasks=set(t)) for t in [
            ('ns_spr',),
            erp_tasks + ('ns_spr',),
            ns_froi_tasks + ('ns_spr',),
            erp_tasks + ns_froi_tasks,
            erp_tasks + ns_froi_tasks + ('ns_spr',)]]
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
            loss_tasks=set(ns_froi_tasks))
        settings.head_graph_parts[ResponseKind.ns_froi] = OrderedDict(
            untransformed_pooled_linear=KeyedLinear(
                ('bert', 'untransformed_pooled'), is_sequence=False, apply_at_most_one_data_id='if_no_target',
                targets=ResponseKind.ns_froi))
        return settings
    elif name == 'number_agreement':
        agr = ('colorless', 'linzen_agree')
        return [Settings(
            corpora=(
                corpus_types.ColorlessGreenCorpus(), corpus_types.LinzenAgreementCorpus(), corpus_types.UclCorpus()),
            optimization_settings=OptimizationSettings(num_train_epochs=50),
            num_runs=10,
            loss_tasks=set(t)) for t in [agr, erp_tasks + agr, erp_tasks]]
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
                ('bert', 'untransformed_pooled'), is_sequence=False, apply_at_most_one_data_id='if_no_target',
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
            batch_kind='single_task_random',
            num_runs=4)
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True)]
        settings.head_graph_parts[ResponseKind.hp_fmri] = OrderedDict(
            untransformed_pooled_linear=KeyedLinear(
                ('bert', 'untransformed_pooled'), is_sequence=False, apply_at_most_one_data_id='if_no_target',
                targets=ResponseKind.hp_fmri))
        return settings
    elif name == 'hp_fmri_meg_joint':
        settings = Settings(
            corpora=(corpus_types.HarryPotterCorpus(
                fmri_subjects=hp_fmri_subjects,
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
                ('bert', 'untransformed_pooled'), is_sequence=False, apply_at_most_one_data_id='if_no_target',
                targets=ResponseKind.hp_fmri, force_cpu=True))
        settings.head_graph_parts[ResponseKind.hp_meg] = OrderedDict(
            sequence_linear=KeyedLinear(
                ('bert', 'sequence'), is_sequence=True, targets=ResponseKind.hp_meg, force_cpu=True))
        settings.loss_tasks = set(hp_fmri_tasks + ('hp_meg',))
        return settings
    elif name in ['hp_fmri_simple_{}'.format(s) for s in hp_fmri_subjects]:
        subject_ = name[name.var.rindex('_') + 1:]
        settings = Settings(
            corpora=(corpus_types.HarryPotterCorpus(
                fmri_subjects='hp_fmri_{}'.format(subject_),
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
                ('bert', 'untransformed_pooled'), is_sequence=False, apply_at_most_one_data_id='if_no_target',
                targets=ResponseKind.hp_fmri))
        return settings
    elif name == 'hp_fmri_simple':
        return OrderedDict(
            ('hp_fmri_simple_{}'.format(s), _named_variations('hp_fmri_simple_{}'.format(s))) for s in hp_fmri_subjects)
    elif name == 'sst':
        return Settings(corpora=(corpus_types.StanfordSentimentTreebank(),), loss_tasks={'sentiment'}, num_runs=1)
    elif name == 'hp_from_I':
        settings = Settings(
            corpora=(corpus_types.HarryPotterCorpus(
                fmri_subjects=hp_fmri_subjects,
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
                ('bert', 'untransformed_pooled'), is_sequence=False, apply_at_most_one_data_id='if_no_target',
                targets=ResponseKind.hp_fmri))
        training_variations = list()
        for s in hp_fmri_subjects:
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
                fmri_subjects=hp_fmri_subjects,
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
                ('bert', 'untransformed_pooled'), is_sequence=False, apply_at_most_one_data_id='if_no_target',
                targets=ResponseKind.hp_fmri, hidden_sizes=[20]))
        settings.critics[ResponseKind.hp_fmri] = CriticSettings(critic_type=CriticKeys.single_binary_cross_entropy)
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
                meg_subjects=hp_meg_subjects),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=100,
                num_epochs_train_prediction_heads_only=50,
                num_final_epochs_train_prediction_heads_only=0),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg),
            num_runs=4,
            loss_tasks=set('hp_meg_{}'.format(s) for s in hp_meg_subjects))
        settings.preprocessors[ResponseKind.hp_meg] = [
            PreprocessQuantileDigitize(
                quantiles=2,
                stop_mode='content',
                metadata_example_group_by='fmri_runs',
                train_on_all=True,
                use_one_hot=False)
        ]
        settings.critics[ResponseKind.hp_meg] = CriticSettings(
            critic_type=CriticKeys.cross_entropy, critic_kwargs=dict(num_classes=2))
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
                meg_subjects=hp_meg_subjects),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=100,
                num_epochs_train_prediction_heads_only=50,
                num_final_epochs_train_prediction_heads_only=0),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg),
            batch_kind=('single_task_uniform', 100),
            weight_losses_by_inverse_example_counts=False,
            num_runs=4,
            loss_tasks=set('hp_meg_{}'.format(s) for s in hp_meg_subjects))
        settings.preprocessors[ResponseKind.hp_meg] = [
            PreprocessQuantileDigitize(
                quantiles=2,
                stop_mode='content',
                metadata_example_group_by='fmri_runs',
                train_on_all=True,
                use_one_hot=False)
        ]
        settings.critics[ResponseKind.hp_meg] = CriticSettings(
            critic_type=CriticKeys.cross_entropy, critic_kwargs=dict(num_classes=2))
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
                meg_subjects=hp_meg_subjects),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=1000,
                num_epochs_train_prediction_heads_only=100,
                num_final_epochs_train_prediction_heads_only=0),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg),
            weight_losses_by_inverse_example_counts=True,
            num_runs=4,
            loss_tasks=set('hp_meg_{}'.format(s) for s in hp_meg_subjects))
        settings.preprocessors[ResponseKind.hp_meg] = [
            PreprocessQuantileDigitize(
                quantiles=2,
                stop_mode='content',
                metadata_example_group_by='fmri_runs',
                train_on_all=True,
                use_one_hot=False)
        ]

        settings.head_graph_parts[ResponseKind.hp_meg] = OrderedDict(meg_linear=KeyedLinear(
                ('bert', 'sequence', 'all'), is_sequence=True,
                hidden_sizes=[100], hidden_activation=None, targets=ResponseKind.hp_meg))

        settings.critics[ResponseKind.hp_meg] = CriticSettings(
            critic_type=CriticKeys.cross_entropy, critic_kwargs=dict(num_classes=2))
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
                meg_subjects=hp_meg_subjects),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=1,
                num_epochs_train_prediction_heads_only=0,
                num_final_epochs_train_prediction_heads_only=0),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg),
            batch_kind=('single_task_uniform', 100),
            weight_losses_by_inverse_example_counts=False,
            loss_tasks=set('hp_meg_{}'.format(s) for s in hp_meg_subjects),
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
                    ('bert', 'sequence', 'all'), is_sequence=True,
                    output_key_to_shape=OrderedDict(sequence_all_bottleneck=10),
                    should_norm=True))))

        settings.head_graph_parts[ResponseKind.hp_meg] = OrderedDict(meg_linear=KeyedLinear(
            'sequence_all_bottleneck', is_sequence=True, targets=ResponseKind.hp_meg))

        settings.critics[ResponseKind.hp_meg] = CriticSettings(
            critic_type=CriticKeys.cross_entropy, critic_kwargs=dict(num_classes=2))
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
            meta_learn_gradient_loss_tasks=set('hp_fmri_{}'.format(s) for s in hp_fmri_subjects),
            num_meta_learn_gradient_samples=10,
            num_meta_learn_no_gradient_samples=0,
            weight_losses_by_inverse_example_counts=False,
            batch_kind=('single_task_uniform', 100),
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
                ('bert', 'untransformed_pooled'), is_sequence=False, apply_at_most_one_data_id='if_no_target',
                targets=ResponseKind.hp_fmri))
        settings.critics[ResponseKind.hp_fmri] = CriticSettings(
            critic_type=CriticKeys.single_cross_entropy, critic_kwargs=dict(num_classes=2))
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
            meta_learn_gradient_loss_tasks=set('hp_fmri_{}'.format(s) for s in hp_fmri_subjects).union(
                'hp_meg_{}'.format(s) for s in hp_meg_subjects),
            num_meta_learn_gradient_samples=10,
            num_meta_learn_no_gradient_samples=0,
            weight_losses_by_inverse_example_counts=False,
            batch_kind=('single_task_uniform', 100),
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
                        ('bert', 'sequence', 'all'), is_sequence=True,
                        output_key_to_shape=OrderedDict(sequence_all_bottleneck=10),
                        should_norm=True))),
            pooled_bottleneck=PooledFromSequence('sequence_all_bottleneck', 'pooled_all_bottleneck'))
        settings.head_graph_parts[ResponseKind.hp_meg] = OrderedDict(meg_linear=KeyedLinear(
            'sequence_all_bottleneck', is_sequence=True, targets=ResponseKind.hp_meg))
        settings.head_graph_parts[ResponseKind.hp_fmri] = OrderedDict(fmri_linear=KeyedLinear(
            'pooled_all_bottleneck',
            is_sequence=False,
            apply_at_most_one_data_id='if_no_target',
            targets=ResponseKind.hp_fmri))
        settings.critics[ResponseKind.hp_meg] = CriticSettings(
            critic_type=CriticKeys.cross_entropy, critic_kwargs=dict(num_classes=2))
        settings.critics[ResponseKind.hp_fmri] = CriticSettings(
            critic_type=CriticKeys.single_cross_entropy, critic_kwargs=dict(num_classes=2))
        return settings
    elif name == 'superglue':
        settings = Settings(
            corpora=(
                corpus_types.BooleanQuestions(),
                corpus_types.CommitmentBank(),
                corpus_types.ChoiceOfPlausibleAlternatives(),
                # corpus_types.MultiSentenceReadingComprehension(),
                # corpus_types.ReadingComprehensionWithCommonSenseReasoning(),
                corpus_types.RecognizingTextualEntailment(),
                corpus_types.WinogradSchemaChallenge(),
                corpus_types.WordInContext()),
            optimization_settings=OptimizationSettings(
                num_train_epochs=10,
                num_epochs_train_prediction_heads_only=0,
                num_final_epochs_train_prediction_heads_only=0,
                learning_rate=1e-5,
                learning_rate_schedule=LearningRateSchedule('linear_warmup_rsqrt_decay', num_warmup_steps=400),
                train_batch_size=8,
                predict_batch_size=8),
            loss_tasks=set(),
            weight_losses_by_inverse_example_counts=False,
            batch_kind=('single_task_uniform', 5000),
            num_runs=1)
        for corpus in settings.corpora:
            heads, critics = superglue_heads(
                corpus, sequence_key=('bert', 'sequence'), pooled_key=('bert', 'pooled'))
            settings.head_graph_parts.update(heads)
            settings.critics.update(critics)
            for k in critics:
                settings.loss_tasks.add(k)
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
                learning_rate_schedule=LearningRateSchedule('linear_warmup_rsqrt_decay', num_warmup_steps=400),
                train_batch_size=8,
                predict_batch_size=8),
            loss_tasks=set(),
            weight_losses_by_inverse_example_counts=False,
            num_meta_learn_gradient_samples=10,
            num_meta_learn_no_gradient_samples=0,
            batch_kind=('single_task_uniform', 500),
            num_runs=1)
        settings.common_graph_parts = OrderedDict(
            contextual_bottleneck=LinearContextualParameterGeneration(
                'response_id', 'num_response_data_fields', 3,
                OrderedDict(
                    bottleneck=KeyedLinear(
                        ('bert', 'sequence', 'all'), is_sequence=True,
                        output_key_to_shape=OrderedDict(sequence_all_bottleneck=10),
                        should_norm=True))),
            pooled_bottleneck=PooledFromSequence('sequence_all_bottleneck', 'pooled_all_bottleneck'))
        for corpus in settings.corpora:
            heads, critics = superglue_heads(
                corpus, sequence_key='sequence_all_bottleneck', pooled_key='pooled_all_bottleneck')
            settings.head_graph_parts.update(heads)
            settings.critics.update(critics)
            for k in critics:
                settings.meta_learn_gradient_loss_tasks.add(k)
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
                learning_rate_schedule=LearningRateSchedule('linear_warmup_rsqrt_decay', num_warmup_steps=400),
                train_batch_size=8,
                predict_batch_size=8),
            loss_tasks=set(),
            weight_losses_by_inverse_example_counts=False,
            num_meta_learn_gradient_samples=10,
            num_meta_learn_no_gradient_samples=0,
            batch_kind=('single_task_uniform', 500),
            num_runs=1)
        settings.common_graph_parts = OrderedDict(
            contextual_bottleneck=LinearContextualParameterGeneration(
                'response_id', 'num_response_data_fields', 3,
                OrderedDict(
                    bottleneck=KeyedLinear(
                        ('bert', 'sequence', 'all'), is_sequence=True,
                        output_key_to_shape=OrderedDict(sequence_all_bottleneck=10),
                        should_norm=True))),
            pooled_bottleneck=PooledFromSequence('sequence_all_bottleneck', 'pooled_all_bottleneck'))
        for corpus in settings.corpora:
            heads, critics = what_you_can_cram_heads(corpus, pooled_key='pooled_all_bottleneck')
            settings.head_graph_parts.update(heads)
            settings.critics.update(critics)
            for k in critics:
                settings.meta_learn_gradient_loss_tasks.add(k)
        return settings
    else:
        raise ValueError('Unknown name: {}. Valid choices are: \n{}'.format(name.var, '\n'.join(name.tests)))
