import hashlib
import itertools
import random
from collections import OrderedDict
from dataclasses import replace
from typing import Union, Iterable, Mapping, Tuple

import numpy as np
import torch

from .common import SwitchRemember
from .data_sets import ResponseKind, CorpusTypes, PreprocessDetrend, PreprocessStandardize, \
    PreprocessKMeans, PreprocessRandomPair, PreprocessMakeBinary, preprocess_fork_no_cluster_to_disk, \
    PreprocessQuantileDigitize
from .modeling import KeyedLinear, CriticKeys, LinearContextualParameterGeneration
from .settings import Settings, OptimizationSettings, CriticSettings


__all__ = ['task_hash', 'set_random_seeds', 'iterate_powerset', 'named_variations', 'singleton_variation']


def _internal_hash_update(hash_, settings: Settings):
    hash_.update('standard_losses'.encode())
    for loss_task in sorted(settings.loss_tasks):
        hash_.update(loss_task.encode())
    hash_.update('inner_maml_losses'.encode())
    for loss_task in sorted(settings.inner_maml_loss_tasks):
        hash_.update(loss_task.encode())
    hash_.update('outer_maml_losses'.encode())
    for loss_task in sorted(settings.output_maml_loss_tasks):
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
            Settings(corpora=(CorpusTypes.UclCorpus(),), num_runs=100, loss_tasks=set(t))
            for t in iterate_powerset(erp_tasks)]
    elif name == 'erp_joint':
        return Settings(corpora=(CorpusTypes.UclCorpus(),), loss_tasks=set(erp_tasks), num_runs=100)
    elif name == 'nat_stories':
        settings = Settings(
            corpora=(
                CorpusTypes.NaturalStoriesCorpus(
                    froi_window_duration=10.,
                    froi_minimum_duration_required=9.5,
                    froi_use_word_unit_durations=False,
                    froi_sentence_mode='ignore'),
                CorpusTypes.UclCorpus()),
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
                CorpusTypes.NaturalStoriesCorpus(
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
            corpora=(CorpusTypes.ColorlessGreenCorpus(), CorpusTypes.LinzenAgreementCorpus(), CorpusTypes.UclCorpus()),
            optimization_settings=OptimizationSettings(num_train_epochs=50),
            num_runs=10,
            loss_tasks=set(t)) for t in [agr, erp_tasks + agr, erp_tasks]]
    elif name == 'hp_fmri_I_20':
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
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
    elif name == 'hp_fmri_meg_joint':
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
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
            corpora=(CorpusTypes.HarryPotterCorpus(
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
        return Settings(corpora=(CorpusTypes.StanfordSentimentTreebank(),), loss_tasks={'sentiment'}, num_runs=1)
    elif name == 'hp_from_I':
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
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
            corpora=(CorpusTypes.HarryPotterCorpus(
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
                data_id_pair_fn_map=PreprocessRandomPair.pair_from_end)),
            PreprocessMakeBinary(threshold=0)]
        settings.preprocess_fork_fn = preprocess_fork_no_cluster_to_disk
        settings.head_graph_parts[ResponseKind.hp_fmri] = OrderedDict(
            untransformed_pooled_linear=KeyedLinear(
                ('bert', 'untransformed_pooled'), is_sequence=False, apply_at_most_one_data_id='if_no_target',
                targets=ResponseKind.hp_fmri, hidden_sizes=[20]))
        settings.critics[ResponseKind.hp_fmri] = CriticSettings(critic_type=CriticKeys.single_binary_cross_entropy)
        return settings
    elif name == 'hp_meg_diff_drc_25':
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
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
            corpora=(CorpusTypes.HarryPotterCorpus(
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
            corpora=(CorpusTypes.HarryPotterCorpus(
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
            corpora=(CorpusTypes.HarryPotterCorpus(
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
    else:
        raise ValueError('Unknown name: {}. Valid choices are: \n{}'.format(name.var, '\n'.join(name.tests)))
