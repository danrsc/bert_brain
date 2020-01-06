from dataclasses import dataclass, field
from typing import Sequence, Callable, MutableMapping, Mapping, Optional, Union, Tuple, OrderedDict as OrderedDictT

import numpy as np
from .data_sets import PreprocessStandardize, PreprocessLog, \
    PreprocessPCA, PreprocessClip, PreprocessDetrend, HarryPotterMakeLeaveOutFmriRun, PreparedDataView, \
    ResponseKind, InputFeatures, RawData, natural_stories_make_leave_stories_out, KindData, CorpusKeys, CorpusBase, \
    UclCorpus
from .modeling import CriticKeys, GraphPart


__all__ = ['OptimizationSettings', 'CriticSettings', 'Settings']


@dataclass
class OptimizationSettings:
    # Total number of training epochs to perform.
    num_train_epochs: int = 3
    # initial learning rate for Adam
    learning_rate: float = 5e-5
    # Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.
    warmup_proportion: float = 0.1
    train_batch_size: int = 32
    predict_batch_size: int = 8
    # When splitting up a long document into chunks, how much stride to take between chunks.
    doc_stride: int = 128
    # Whether to perform optimization and keep the optimizer averages on CPU
    optimize_on_cpu: bool = False
    # Whether to use 16-bit float precision instead of 32-bit
    fp16: bool = False
    # Loss scaling, positive power of 2 values can improve fp16 convergence.
    loss_scale: float = 128
    # Number of updates steps to accumulate before performing a backward/update pass.
    gradient_accumulation_steps: int = 1
    # local_rank for distributed training on gpus; probably don't need this
    local_rank: int = -1
    # During the first num_epochs_train_prediction_heads_only, only the prediction heads will be trained
    num_epochs_train_prediction_heads_only: int = 0
    # During the last num_final_prediction_head_only_epochs, only the prediction heads will be trained
    num_final_epochs_train_prediction_heads_only: int = 0


@dataclass
class CriticSettings:
    critic_type: str
    critic_kwargs: Optional[Mapping] = None


def _default_split_functions():

    return {
        CorpusKeys.HarryPotterCorpus: HarryPotterMakeLeaveOutFmriRun(),
        CorpusKeys.NaturalStoriesCorpus: natural_stories_make_leave_stories_out
    }


def _default_preprocessors():

    preprocess_standardize = PreprocessStandardize(stop_mode='content')

    # ResponseKind.colorless defaults to None
    # ResponseKind.linzen_agree defaults to None

    # alternate hp_meg for soft_label_cross_entropy
    #     [
    #         #  preprocess_detrend,
    #         #  partial(preprocess_standardize, average_axis=None),
    #         PreprocessDiscretize(bins=np.exp(np.linspace(-0.2, 1., 5))),  # bins=np.arange(6) - 2.5
    #         PreprocessNanMean())]

    return {
        ResponseKind.hp_fmri: [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None)],
        ResponseKind.hp_meg: [
            PreprocessStandardize(average_axis=None, stop_mode='content'),
            PreprocessPCA(stop_mode='content'),
            PreprocessStandardize(average_axis=None, stop_mode='content')],
        # ResponseKind.hp_meg: [
        #     PreprocessDetrend(stop_mode='content', metadata_example_group_by='fmri_runs', train_on_all=True),
        #     PreprocessStandardize(stop_mode='content', average_axis=None)],

        ResponseKind.ucl_erp: preprocess_standardize,
        ResponseKind.ucl_eye: [PreprocessLog(), preprocess_standardize],
        ResponseKind.ucl_self_paced: [PreprocessLog(), preprocess_standardize],
        ResponseKind.ns_reaction_times: [
            PreprocessClip(maximum=3000, value_beyond_max=np.nan), PreprocessLog(), preprocess_standardize],
        ResponseKind.ns_froi: PreprocessStandardize(stop_mode=None),
    }


def _default_supplemental_fields():
    return {'token_lengths', 'token_probabilities'}


def _default_critics():

    return {
        CorpusKeys.UclCorpus: CriticSettings(critic_type=CriticKeys.mse),
        ResponseKind.ns_froi: CriticSettings(critic_type=CriticKeys.single_mse),
        CorpusKeys.NaturalStoriesCorpus: CriticSettings(critic_type=CriticKeys.mse),
        ResponseKind.hp_fmri: CriticSettings(critic_type=CriticKeys.single_mse),
        CorpusKeys.HarryPotterCorpus: CriticSettings(critic_type=CriticKeys.mse),
        CorpusKeys.ColorlessGreenCorpus: CriticSettings(critic_type=CriticKeys.single_binary_cross_entropy),
        CorpusKeys.LinzenAgreementCorpus: CriticSettings(critic_type=CriticKeys.single_binary_cross_entropy),
        CorpusKeys.StanfordSentimentTreebank: CriticSettings(critic_type=CriticKeys.single_binary_cross_entropy)
    }


_PreprocessorTypeUnion = Union[
    Callable[[PreparedDataView, Optional[Mapping[str, np.ndarray]]], PreparedDataView],
    Tuple[
        str,
        Callable[
            [PreparedDataView, Optional[Mapping[str, np.ndarray]]],
            Tuple[PreparedDataView, Optional[Mapping[str, np.ndarray]]]]],
    str]
PreprocessorTypeUnion = Union[_PreprocessorTypeUnion, Sequence[_PreprocessorTypeUnion]]


@dataclass
class Settings:
    # which data to load
    corpora: Optional[Sequence[CorpusBase]] = (UclCorpus(),)

    # maps from a corpus key to a function which takes the index of the current variation and returns another
    # function which will do the splitting. The returned function should take a RawData instance and a RandomState
    # instance and return a tuple of (train, validation, test) examples
    # if not specified, the data will be randomly split according to the way the fields of the RawData instance are set.
    # I.e. if the RawData instance is pre split, that is respected. Otherwise test_proportion and
    # validation_proportion_of_train fields in the RawData instance govern the split
    split_functions: MutableMapping[
        str,
        Callable[[int], Callable[
            [RawData, np.random.RandomState],
            Tuple[
                Optional[Sequence[InputFeatures]],
                Optional[Sequence[InputFeatures]],
                Optional[Sequence[InputFeatures]]]]]] = field(default_factory=_default_split_functions)

    # Mapping from [response_key, kind, or corpus_key] to a preprocessor. Lookups fall back in that
    # order. This determines how the data will be processed. If not specified, no preprocessing is applied
    preprocessors: MutableMapping[str, PreprocessorTypeUnion] = field(default_factory=_default_preprocessors)

    preprocess_fork_fn: Callable[
        [str, KindData, PreprocessorTypeUnion], Tuple[Optional[str], Optional[PreprocessorTypeUnion]]] = None

    bert_model: str = 'bert-base-uncased'
    max_sequence_length: Optional[int] = None
    optimization_settings: OptimizationSettings = OptimizationSettings()

    # fields which should be concatenated with the output of BERT before the prediction heads are applied
    supplemental_fields: set = field(default_factory=_default_supplemental_fields)

    # Graph parts which are applied after BERT but before head_graph_parts. These can be used to apply a common mapping
    # from BERT to a different representation which is then used by head_graph_parts
    common_graph_parts: Optional[OrderedDictT[str, GraphPart]] = None

    # Mapping from [response_key, kind, or corpus_key] to a prediction head. Lookups fall back in that order.
    # This determines how the output from BERT is used to make predictions for each target. Note that whenever two
    # fields map to the same PredictionHeadSettings instance or to two PredictionHeadSettings instances with the same
    # key, the predictions for those fields will be made by a single instance of the prediction head. If two fields map
    # to PredictionHeadSettings instances that have different keys, then the predictions for those fields will be made
    # by two different instances of the prediction head even if the head has the same type. This enables different kinds
    # of parameter-sharing between fields.
    head_graph_parts: MutableMapping[str, OrderedDictT[str, GraphPart]] = field(default_factory=dict)

    # Sequence of [response_key or kind]. Data corresponding to fields specified here will not be put into a
    # batch directly; This allows the system to save significant resources by not padding a tensor for a full
    # sequence of entries when the number of real entries in the tensor is sparse. Mostly, this is for fMRI where
    # each image is huge, and there are relatively few target images relative to the number of tokens in the sequence.
    # If a prediction head modifies the sequence (for example performs a grouping or pooling operation) then that head
    # must also handle modification of the (field, data_ids) so that the alignment between the model predictions and
    # the target is correctly maintained. The system will then go fetch the data corresponding to data_ids
    # and put them into the batch just before the losses are computed.
    data_id_in_batch_keys: Sequence[str] = (ResponseKind.ns_froi, ResponseKind.hp_fmri)

    # Sequence of [response_key or kind]. Data corresponding to fields specified here will not be put into the dataset
    # unless those fields are in the loss
    filter_when_not_in_loss_keys: Optional[Sequence[str]] = None

    # one of:
    # mixed_task_random: Batches contain multiple tasks, all items are visited in an epoch in random order
    # single_task_random: Batches contain a single task, all items are visited in an epoch in random order
    # (single_task_uniform, num_batches_per_epoch):
    #   Batches contain a single task, tasks are uniformly sampled, then items are sampled within
    #   a task to create a batch. num_batches_per_epoch controls epoch length, items may not all be visited in an epoch
    batch_kind: [Union[str, Tuple[str, int]]] = 'mixed_task_random'

    # mapping from [response_key, kind, or corpus_key] to critic settings; lookups fall back in that order
    critics: MutableMapping[str, Union[CriticSettings, str]] = field(default_factory=_default_critics)

    # fields which should be used to evaluate the loss. All critics specified by the critics setting will be invoked,
    # but only the fields listed here will be considered part of the loss for the purpose of optimization and
    # only those fields will be reported in train_loss / eval_loss. Other critic output is available as metrics for
    # reporting, but is otherwise ignored by training.
    loss_tasks: set = field(default_factory=set)

    # Training can use a meta learning mode similar to Reptile (https://arxiv.org/abs/1803.02999) or First-Order
    # Model Agnostic Meta Learning (FOMAML) depending on the precise configuration. To turn on meta-learning mode,
    # set meta_learn_gradient_loss_tasks to non-empty. If meta_learn_gradient_loss_tasks is non-empty, then loss_tasks
    # must be empty. Training works by running num_meta_learn_no_gradient_samples batches of the tasks which appear
    # in meta_learn_no_gradient_loss_tasks, followed by num_meta_learn_gradient_samples of the tasks which appear in
    # meta_learn_gradient_loss_tasks. The meta-gradient is computed by storing the parameter values just before the
    # batches of meta_learn_gradient_loss_tasks and subtracting the parameter values after those batches from the
    # parameter values before those batches. Reptile can be set up by putting all tasks into
    # meta_learn_gradient_loss_tasks. FOMAML can be set up by putting all tasks into both
    # meta_learn_no_gradient_loss_tasks and meta_learn_gradient_loss_tasks and setting num_meta_learn_gradient_samples
    # to 1 and num_meta_learn_no_gradient_samples to k > 1. In the FOMAML framework, the meta_learn_gradient_loss_tasks
    # would be validation tasks (but note that like Reptile, we don't do a train/test split here). We also consider
    # an algorithm inspired by FOMAML which is meant for transfer learning from higher SnR to lower SnR tasks
    # where we put the high SnR tasks into meta_learn_no_gradient_loss_tasks and low SnR tasks into
    # meta_learn_gradient_loss_tasks
    meta_learn_no_gradient_loss_tasks: set = field(default_factory=set)
    meta_learn_gradient_loss_tasks: set = field(default_factory=set)
    num_meta_learn_no_gradient_samples: int = 10
    num_meta_learn_gradient_samples: int = 10

    @property
    def all_loss_tasks(self):
        all_loss_tasks = set(self.loss_tasks)
        all_loss_tasks.update(self.meta_learn_no_gradient_loss_tasks)
        all_loss_tasks.update(self.meta_learn_gradient_loss_tasks)
        return all_loss_tasks

    # when specified, this acts as a key to determine a partially pre-trained model which should be used as a starting
    # point
    load_from: str = None

    # Used in combination with load from to map the run of the current model to a run of the model we're loading
    # If None, uses identity mapping
    load_from_run_map: Optional[Callable[[int], int]] = None

    # If true, tasks are re-weighted by the inverse of the number of available examples so that the expected
    # effect of different data sets is balanced
    weight_losses_by_inverse_example_counts: bool = True

    # fields which are not in the response_data part of the RawData structure, but which should nevertheless be used
    # as output targets of the model. If a field is already in loss_tasks then it does not need to also be specified
    # here, but this allows non-loss fields to be output/evaluated. This is useful when we want to predict something
    # like syntax that is on the input features rather than in the response data. The use-case for this is a model
    # has been pre-trained to make a prediction on a non-response field, and we want to track how those predictions
    # are changing as we target a different optimization metric.
    non_response_outputs: set = field(default_factory=set)

    seed: int = 42

    # The number of times to run the training
    num_runs: int = 1

    # turn on tqdm at more granular levels
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

    @property
    def is_one_task_at_a_time(self):
        if isinstance(self.batch_kind, (tuple, list)):
            return self.batch_kind[0].startswith('single_task_')
        else:
            return self.batch_kind.startswith('single_task_')
