from dataclasses import dataclass, field
from typing import Any, Sequence, Callable, MutableMapping, Mapping, Optional, Union, Tuple

import numpy as np
from bert_erp_datasets import DataKeys, PreprocessMany, PreprocessStandardize, PreprocessLog, \
    PreprocessPCA, PreprocessClip, PreprocessDetrend, harry_potter_make_leave_out_fmri_run, PreparedDataView, \
    ResponseKind, InputFeatures, RawData, natural_stories_make_leave_stories_out
from bert_erp_modeling import CriticKeys


__all__ = ['CriticSettings', 'Settings']


@dataclass
class CriticSettings:
    critic_type: str
    critic_kwargs: Optional[Mapping] = None
    # stop_word_mode can be:
    #     None: no distinction made between content words and stop words
    #     'train_both': training is done on both, results are reported separately
    #     'train_content': training is done on content, both results reported
    #     'report_content': training is done on content, only content results reported
    stop_word_mode: Optional[str] = 'train_content'


def _default_preprocessors():

    preprocess_standardize = PreprocessStandardize(stop_mode='content')

    # ResponseKind.colorless defaults to None
    # ResponseKind.linzen_agree defaults to None

    # alternate hp_meg for soft_label_cross_entropy
    #     PreprocessMany(
    #         #  preprocess_detrend,
    #         #  partial(preprocess_standardize, average_axis=None),
    #         PreprocessDiscretize(bins=np.exp(np.linspace(-0.2, 1., 5))),  # bins=np.arange(6) - 2.5
    #         PreprocessNanMean()))

    return {
        ResponseKind.hp_fmri: PreprocessMany(
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode='content')),
        ResponseKind.hp_meg: PreprocessMany(
            PreprocessStandardize(average_axis=None, stop_mode='content'),
            PreprocessPCA(stop_mode='content'),
            PreprocessStandardize(average_axis=None, stop_mode='content')),
        ResponseKind.ucl_erp: preprocess_standardize,
        ResponseKind.ucl_eye: PreprocessMany(PreprocessLog(), preprocess_standardize),
        ResponseKind.ucl_self_paced: PreprocessMany(PreprocessLog(), preprocess_standardize),
        ResponseKind.ns_reaction_times: PreprocessMany(
            PreprocessClip(maximum=3000, value_beyond_max=np.nan), PreprocessLog(), preprocess_standardize),
        ResponseKind.ns_froi: preprocess_standardize,
    }


def _default_critics():

    return {
        DataKeys.ucl: CriticSettings(critic_type=CriticKeys.mse),
        DataKeys.natural_stories: CriticSettings(critic_type=CriticKeys.mse),
        ResponseKind.hp_fmri: CriticSettings(critic_type=CriticKeys.pooled_mse),
        DataKeys.harry_potter: CriticSettings(critic_type=CriticKeys.mse),
        DataKeys.colorless_green: CriticSettings(critic_type=CriticKeys.pooled_binary_cross_entropy),
        DataKeys.linzen_agreement: CriticSettings(critic_type=CriticKeys.pooled_binary_cross_entropy)
    }


def _default_split_functions():

    return {
        DataKeys.harry_potter: harry_potter_make_leave_out_fmri_run,
        DataKeys.natural_stories: natural_stories_make_leave_stories_out
    }


def _default_data_key_kwargs():
    return {
        DataKeys.ucl: dict(include_erp=True, include_eye=True, self_paced_inclusion='eye'),
        DataKeys.harry_potter: dict(fmri_subjects='I')}


def _default_additional_fields():
    return {'token_lengths', 'token_probabilities'}


@dataclass
class Settings:
    # can use an extension to render to file, e.g. 'png', or 'show' to plot
    visualize_mode: str = None

    # which data to load
    task_data_keys: Optional[Sequence[str]] = (DataKeys.ucl,)

    # keyword args for loading data
    data_key_kwargs: Optional[Mapping[str, Mapping[str, Any]]] = field(default_factory=_default_data_key_kwargs)

    # mapping from [response_key, kind, or corpus_key] to critic settings; lookups fall back in that order
    critics: MutableMapping[str, Union[CriticSettings, str]] = field(default_factory=_default_critics)

    preprocessors: MutableMapping[
            str, Callable[[PreparedDataView, Optional[Mapping[str, np.array]]], PreparedDataView]] = \
        field(default_factory=_default_preprocessors)

    split_functions: MutableMapping[
        str,
        Callable[[int], Callable[
            [RawData, np.random.RandomState],
            Tuple[
                Optional[Sequence[InputFeatures]],
                Optional[Sequence[InputFeatures]],
                Optional[Sequence[InputFeatures]]]]]] = field(default_factory=_default_split_functions)

    # these keys will be grouped by data_id so if more than one word in a sequence maps to the same image, the
    # predictions of each word mapping to a single image will be averaged before the linear layer is applied
    grouped_prediction_keys: Optional[Sequence[str]] = (ResponseKind.hp_fmri, ResponseKind.ns_froi)

    bert_model: str = 'bert-base-uncased'

    # Total number of training epochs to perform.
    num_train_epochs: float = 3.0

    # initial learning rate for Adam
    learning_rate: float = 5e-5

    # Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.
    warmup_proportion: float = 0.1

    train_batch_size: int = 32

    predict_batch_size: int = 8

    # When splitting up a long document into chunks, how much stride to take between chunks.
    doc_stride: int = 128

    # fields which should be used to evaluate the loss
    loss_tasks: set = field(default_factory=set)

    # fields which are not in the response_data part of the RawData structure, but which should nevertheless be used
    # as output targets of the model. If a field is already in loss_tasks then it does not need to also be specified
    # here, but this allows non-loss fields to be output/evaluated
    non_response_outputs: set = field(default_factory=set)

    # fields which should be concatenated with the output of BERT before the head is applied
    additional_input_fields: set = field(default_factory=_default_additional_fields)

    save_checkpoints_steps: int = 1000

    seed: int = 42

    # Number of updates steps to accumulate before performing a backward/update pass.
    gradient_accumulation_steps: int = 1

    # local_rank for distributed training on gpus; probably don't need this
    local_rank: int = -1

    # Whether to perform optimization and keep the optimizer averages on CPU
    optimize_on_cpu: bool = False

    # Whether to use 16-bit float precision instead of 32-bit
    fp16: bool = False

    # Loss scaling, positive power of 2 values can improve fp16 convergence.
    loss_scale: float = 128

    # The total number of n-best predictions to generate in the nbest_predictions.json output file.
    n_best_size: int = 20

    verbose_logging: bool = False

    show_epoch_progress: bool = False

    show_step_progress: bool = False

    # Whether not to use CUDA when available
    no_cuda: bool = False

    def get_split_functions(self, index_run):
        result = dict()
        for key in self.split_functions:
            if self.split_functions[key] is not None:
                result[key] = self.split_functions[key](index_run)
        if len(result) == 0:
            return None
        return result

    def _lookup_critic(self, field_name, data_set):
        if field_name in self.critics:
            return self.critics[field_name]
        if data_set.is_response_data(field_name):
            kind = data_set.response_data_kind(field_name)
            if kind in self.critics:
                return self.critics[kind]
            data_key = data_set.data_set_key_for_field(field_name)
            if data_key in self.critics:
                return self.critics[data_key]
        return CriticKeys.mse

    def get_critic(self, field_name, data_set):
        critic = self._lookup_critic(field_name, data_set)
        if isinstance(critic, str):
            return CriticSettings(critic)
        return critic
