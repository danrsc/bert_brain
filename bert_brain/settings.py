from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence, Callable, MutableMapping, Mapping, \
    Optional, Union, Tuple, OrderedDict as OrderedDictT, Any
from inspect import signature

import numpy as np
from torch.optim.optimizer import Optimizer

from transformers import AdamW

from .data_sets import PreprocessStandardize, PreprocessLog, \
    PreprocessPCA, PreprocessClip, PreprocessDetrend, HarryPotterMakeLeaveOutFmriRun, \
    ResponseKind, NaturalStoriesMakeLeaveStoriesOut, corpus_types, CorpusBase, \
    UclCorpus, PreprocessorSequenceT, PreprocessForkFnT, SplitFunctionT, SamplerFactory, RandomSamplerFactory
from .modeling import critic_types, GraphPartFactory, NamedTargetMaskedLossBase, LearningRateScheduleFactory, \
    learning_rate_schedules


__all__ = ['OptimizationSettings', 'Settings', 'make_optimizer_factory', 'make_inner_meta_learn_optimizer_factory',
           'ParameterGroupOptimizationSettings']


def make_optimizer_factory(optimizer_type: type, **kwargs) -> Callable[[Iterable[Mapping[str, Any]], float], Optimizer]:
    def make_optimizer(params, lr):
        return optimizer_type(params=params, lr=lr, **kwargs)

    return make_optimizer


def make_inner_meta_learn_optimizer_factory(
        optimizer_type: type,
        use_outer_optimizer_as_defaults: bool = True,
        partial_sequence_overrides: Optional[Union[Iterable[str], str]] = 'betas',
        **kwargs) -> Callable[[Optimizer, Iterable[Mapping[str, Any]], float], Optimizer]:
    def make_optimizer(outer_optimizer: Optimizer, params, lr):
        kwargs_ = kwargs
        if use_outer_optimizer_as_defaults:
            partial_sequence_overrides_ = partial_sequence_overrides \
                if np.ndim(partial_sequence_overrides) > 0 else (partial_sequence_overrides,)
            # noinspection PyUnresolvedReferences
            outer_kwargs = dict(
                (k, outer_optimizer.defaults[k])
                for k in signature(optimizer_type.__init__).parameters if k in outer_optimizer.defaults and k != 'lr')
            for key in partial_sequence_overrides_:
                if key in outer_kwargs and key in kwargs_:
                    if np.ndim(outer_kwargs[key]) != np.ndim(kwargs_[key]):
                        raise ValueError('Incompatible partial sequence overrides: {}'.format(key))
                    if np.ndim(outer_kwargs[key]) == 0:
                        # scalar, just do normal override
                        continue
                    if len(outer_kwargs[key]) != len(kwargs_[key]):
                        raise ValueError('Incompatible partial sequence overrides: {}'.format(key))
                    kwargs_[key] = type(outer_kwargs[key])(
                        outer if inner is None else inner for outer, inner in zip(
                            outer_kwargs[key], kwargs_[key]))
            outer_kwargs.update(kwargs_)
            kwargs_ = outer_kwargs
        return optimizer_type(params=params, lr=lr, **kwargs_)

    return make_optimizer


class ParameterGroupOptimizationSettings:

    def __init__(self, learning_rate=5e-5, num_inactive_start_epochs=0, num_inactive_end_epochs=0, **kwargs):
        self.learning_rate = learning_rate
        self.num_inactive_start_epochs = num_inactive_start_epochs
        self.num_inactive_end_epochs = num_inactive_end_epochs
        self.kwargs = kwargs


ParameterGroupSettingsT = Union[ParameterGroupOptimizationSettings, Mapping[str, ParameterGroupOptimizationSettings]]


@dataclass
class OptimizationSettings:
    # Total number of training epochs to perform.
    num_train_epochs: int = 3
    make_optimizer: Callable[[Iterable[Mapping[str, Any]], float], Optimizer] = make_optimizer_factory(
        AdamW, correct_bias=False)
    make_inner_meta_learn_optimizer: Callable[[Optimizer, Iterable[Mapping[str, Any]], float], Optimizer] = \
        make_inner_meta_learn_optimizer_factory(AdamW, betas=(0, None))

    parameter_group_settings: ParameterGroupSettingsT = ParameterGroupOptimizationSettings()

    learning_rate_schedule: LearningRateScheduleFactory = \
        learning_rate_schedules.LinearWithWarmupLearningRateScheduleFactory()
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
    # Passed as num_workers to torch.utils.DataLoader
    num_loader_workers: int = 0

    def get_parameter_group_settings(self, candidate_name):
        if isinstance(self.parameter_group_settings, ParameterGroupOptimizationSettings):
            return 'default', self.parameter_group_settings
        if candidate_name in self.parameter_group_settings:
            return candidate_name, self.parameter_group_settings[candidate_name]
        if candidate_name.endswith('_head') and 'head' in self.parameter_group_settings:
            return 'head', self.parameter_group_settings['head']
        return 'default', self.parameter_group_settings['default']


def _default_split_functions():

    return {
        corpus_types.HarryPotterCorpus.__name__: HarryPotterMakeLeaveOutFmriRun(),
        corpus_types.NaturalStoriesCorpus.__name__: NaturalStoriesMakeLeaveStoriesOut()
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

    result = {
        corpus_types.UclCorpus.__name__: critic_types.NamedTargetStopWordAwareMSE(),
        ResponseKind.ns_froi: critic_types.NamedTargetSingleMSE(),
        corpus_types.NaturalStoriesCorpus.__name__: critic_types.NamedTargetStopWordAwareMSE(),
        ResponseKind.hp_fmri: critic_types.NamedTargetSingleMSE(),
        corpus_types.HarryPotterCorpus.__name__: critic_types.NamedTargetStopWordAwareMSE()
    }

    for corpus_type_str in corpus_types.__all__:
        corpus_type = getattr(corpus_types, corpus_type_str)
        if hasattr(corpus_type, 'num_classes'):
            is_sequence_labeled = False
            if hasattr(corpus_type, 'is_sequence_labeled'):
                is_sequence_labeled = corpus_type.is_sequence_labeled()
            if is_sequence_labeled:
                if corpus_type.num_classes() > 2:
                    result[corpus_type.__name__] = critic_types.NamedTargetStopWordAwareCrossEntropy(
                        num_classes=corpus_type.num_classes())
                else:
                    result[corpus_type.__name__] = critic_types.NamedTargetStopWordAwareBinaryCrossEntropyWithLogits()
            else:
                if corpus_type.num_classes() > 2:
                    result[corpus_type.__name__] = critic_types.NamedTargetSingleCrossEntropy(
                        num_classes=corpus_type.num_classes())
                else:
                    result[corpus_type.__name__] = critic_types.NamedTargetSingleBinaryCrossEntropyWithLogits()

    return result


class MixedTask:

    def __init__(self, alternative_probabilities):
        probabilities = list()
        alternatives = list()
        for alternative, prob in alternative_probabilities:
            probabilities.append(prob)
            alternatives.append(alternative)
        self.probabilities = np.array(probabilities)
        self.alternatives = np.array(alternatives)
        self.probabilities.setflags(write=False)
        self.alternatives.setflags(write=False)
        if np.sum(self.probabilities) != 1:
            raise ValueError('probabilities must be == 1')

    def label(self):
        return np.random.choice(self.alternatives, p=self.probabilities).item()

    @staticmethod
    def make_symmetric_mixed_tasks(alternatives, primary_proportion):
        alternate_proportions = (1 - primary_proportion) / (len(alternatives) - 1)
        result = dict()
        for i in range(len(alternatives)):
            primary = alternatives[i]
            alternates = alternatives[:i] + alternatives[i+1:]
            result[primary] = MixedTask(
                [(primary, primary_proportion)] + [(a, alternate_proportions) for a in alternates])
        return result


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
    split_functions: MutableMapping[str, Callable[[int], SplitFunctionT]] = field(
        default_factory=_default_split_functions)

    # Mapping from [response_key, kind, or corpus_key] to a preprocessor. Lookups fall back in that
    # order. This determines how the data will be processed. If not specified, no preprocessing is applied
    preprocessors: MutableMapping[str, PreprocessorSequenceT] = field(default_factory=_default_preprocessors)

    preprocess_fork_fn: Optional[PreprocessForkFnT] = None

    bert_model: str = 'bert-base-uncased'
    max_sequence_length: Optional[int] = None
    optimization_settings: OptimizationSettings = OptimizationSettings()

    # fields which should be concatenated with the output of BERT before the prediction heads are applied
    supplemental_fields: set = field(default_factory=_default_supplemental_fields)

    # Graph parts which are applied after BERT but before head_graph_parts. These can be used to apply a common mapping
    # from BERT to a different representation which is then used by head_graph_parts
    common_graph_parts: Optional[OrderedDictT[str, GraphPartFactory]] = None

    # Mapping from [response_key, kind, or corpus_key] to a prediction head. Lookups fall back in that order.
    # This determines how the output from BERT is used to make predictions for each target. Note that whenever two
    # fields map to the same PredictionHeadSettings instance or to two PredictionHeadSettings instances with the same
    # key, the predictions for those fields will be made by a single instance of the prediction head. If two fields map
    # to PredictionHeadSettings instances that have different keys, then the predictions for those fields will be made
    # by two different instances of the prediction head even if the head has the same type. This enables different kinds
    # of parameter-sharing between fields.
    head_graph_parts: MutableMapping[str, OrderedDictT[str, GraphPartFactory]] = field(default_factory=dict)

    # Default prediction heads will use these locations in the graph as the source
    default_sequence_source: Union[str, Tuple[str, ...]] = ('bert', 'sequence')
    default_pooled_source: Union[str, Tuple[str, ...]] = ('bert', 'pooled')

    # Sequence of [response_key or kind]. Data corresponding to fields specified here will not be put into a
    # batch directly; This allows the system to save significant resources by not padding a tensor for a full
    # sequence of entries when the number of real entries in the tensor is sparse. Mostly, this is for fMRI where
    # each image is huge, and there are relatively few target images relative to the number of tokens in the sequence.
    # If a prediction head modifies the sequence (for example performs a grouping or pooling operation) then that head
    # must also handle modification of the (field, data_ids) so that the alignment between the model predictions and
    # the target is correctly maintained. The system will then go fetch the data corresponding to data_ids
    # and put them into the batch just before the losses are computed.
    data_id_in_batch_keys: Optional[Sequence[str]] = (ResponseKind.ns_froi, ResponseKind.hp_fmri)

    # Sequence of [response_key or kind]. Data corresponding to fields specified here will not be put into the dataset
    # unless those fields are in the loss
    filter_when_not_in_loss_keys: Optional[Sequence[str]] = None

    # Optional mapping from a [response_key, kind, or corpus_key] to a dictionary which is called as:
    #   replacers = field_spec_replacers[response_key]
    #   field_spec = dataclasses.replace(field_spec, **replacers)
    # to override specific attributes of a FieldSpec. For example, this can be used to modify the tensor type
    # or to treat a sequence as a non-sequence (when only when data_id is valid)
    field_spec_replacers: Optional[Mapping[str, Mapping[str, Any]]] = None

    # An instances of a SamplerFactory which will be used to create the training sampler.
    # If sampler_on_evaluate_factory is None, then during evaluate a BatchOneTaskEvalSampler with
    # maximum 100 samples per task (500 when return_detailed=True) is used whenever sampler_factory generates a one
    # task sampler and a RandomSampler with 10000 samples is used (50000 when return_detailed=True) whenever
    # sampler_factory generates a mixed task sampler. If sampler_on_evaluate_factory is set to 'train', then
    # sampler_factory is used to generate the evaluation sampler. If sampler_on_evaluate_factory is set to a
    # SamplerFactory then that factory will be used to generate the evaluation sampler.
    sampler_factory: SamplerFactory = RandomSamplerFactory()
    sampler_on_evaluate_factory: Optional[Union[SamplerFactory, str]] = None

    # mapping from [response_key, kind, or corpus_key] to critic settings; lookups fall back in that order
    critics: MutableMapping[str, NamedTargetMaskedLossBase] = field(default_factory=_default_critics)

    # fields which should be used to evaluate the loss. All critics specified by the critics setting will be invoked,
    # but only the fields listed here will be considered part of the loss for the purpose of optimization and
    # only those fields will be reported in train_loss / eval_loss. Other critic output is available as metrics for
    # reporting, but is otherwise ignored by training.
    loss_tasks: set = field(default_factory=set)

    # fields listed here are sometimes renamed to a different label to simulate tasks being combined together
    # in different proportions. This enables exploration of how similarity of tasks changes as a function of
    # the mixing proportions
    mixed_tasks: Optional[Mapping[str, MixedTask]] = None

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

    create_meta_train_dataset: bool = False

    # If True, then the meta-learning strategy is changed to use projected conflicting gradients as in
    # Gradient Surgery for Multi-Task Learning
    # Yu, Tianhe et al.
    # https://arxiv.org/pdf/2001.06782.pdf
    # In this case, gradients are computed for num_meta_learn_gradient_samples without any steps being taken.
    # The gradients are temporarily stored, then gradients are adjusted based on cosine similarity of the gradients
    # across tasks by projecting one gradient onto another when the similarity is less than 0
    # When use_pc_grad is True, num_meta_learn_no_gradient_samples must be 0, and meta_learn_no_gradient_loss_tasks must
    # be empty
    use_pc_grad: bool = False

    update_individual_heads_on_dds_sample: bool = False

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
    weight_losses_fn: Optional[Callable[[Mapping[Tuple[str, str], int]], Mapping[str, float]]] = None

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

    # Whether not to use CUDA when available
    no_cuda: bool = False

    def get_split_function(self, corpus, index_run):
        corpus = replace(corpus, index_run=index_run)
        if self.split_functions is None or corpus.corpus_key not in self.split_functions:
            return None
        return self.split_functions[corpus.corpus_key](corpus.run_info)

    def get_critic(self, field_name, data_set):
        if field_name in self.critics:
            return replace(self.critics[field_name], field=field_name)
        if data_set.is_response_data(field_name):
            kind = data_set.response_data_kind(field_name)
            if kind in self.critics:
                return replace(self.critics[kind], field=field_name)
            data_key = data_set.data_set_key_for_field(field_name)
            if data_key in self.critics:
                return replace(self.critics[data_key], field=field_name)
        return critic_types.NamedTargetStopWordAwareMSE(field=field_name)
