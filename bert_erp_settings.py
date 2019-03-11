from dataclasses import dataclass, field
from typing import Any, Sequence, Callable, MutableMapping, Mapping, Optional

import numpy as np
from bert_erp_datasets import DataLoader, DataKeys, PreprocessMany, PreprocessStandardize, PreprocessLog, \
    PreprocessPCA, PreprocessMakeBinary, PreprocessNanMean, PreprocessClip, PreprocessDiscretize, PreprocessDetrend, \
    PreprocessGaussianBlur, harry_potter_split_by_run
from bert_erp_modeling import CriticKeys


__all__ = ['TaskSettings', 'Settings']


@dataclass
class TaskSettings:
    critic_type: str
    critic_kwargs: Optional[Mapping] = None
    preprocessor: Optional[Callable] = None
    # stop_word_mode can be:
    #     None: no distinction made between content words and stop words
    #     'train_both': training is done on both, results are reported separately
    #     'train_content': training is done on content, both results reported
    #     'report_content': training is done on content, only content results reported
    stop_word_mode: Optional[str] = 'train_content'
    split_function: Optional[Callable] = None


def _harry_potter_meg_critic_and_preprocess():

    # return (
    #     CriticKeys.soft_label_cross_entropy,
    #     PreprocessMany(
    #         #  preprocess_detrend,
    #         #  partial(preprocess_standardize, average_axis=None),
    #         PreprocessDiscretize(bins=np.exp(np.linspace(-0.2, 1., 5))),  # bins=np.arange(6) - 2.5
    #         PreprocessNanMean()))

    return (
        CriticKeys.mse,
        PreprocessMany(
            PreprocessStandardize(average_axis=None, stop_mode='content'),
            PreprocessPCA(stop_mode='content'),
            PreprocessStandardize(average_axis=None, stop_mode='content')))


def _default_task_settings():

    # take boxcox transform of these
    ucl_log_keys = {'first_fixation', 'first_pass', 'right_bounded', 'go_past', 'reading_time'}

    preprocess_standardize = PreprocessStandardize(stop_mode='content')

    return {
        DataKeys.ucl: TaskSettings(
            critic_type=CriticKeys.mse,
            preprocessor=PreprocessMany(
                PreprocessLog(data_key_whitelist=ucl_log_keys),
                preprocess_standardize)),
        DataKeys.natural_stories: TaskSettings(
            critic_type=CriticKeys.mse,
            preprocessor=PreprocessMany(
                PreprocessClip(maximum=3000, value_beyond_max=np.nan),
                PreprocessLog(),
                preprocess_standardize)),
        DataKeys.harry_potter_meg: _harry_potter_meg_critic_and_preprocess(),
        DataKeys.harry_potter_fmri: TaskSettings(
            critic_type=CriticKeys.mse,
            preprocessor=PreprocessMany(
                PreprocessStandardize(average_axis=None, metadata_response_group_by='example_run'),
                PreprocessDetrend(stop_mode=None, metadata_response_group_by='example_run')),
            split_function=harry_potter_split_by_run),
        DataKeys.colorless_green: TaskSettings(
            critic_type=CriticKeys.sequence_binary_cross_entropy),
        DataKeys.linzen_agreement: TaskSettings(
            critic_type=CriticKeys.sequence_binary_cross_entropy)
    }


def _default_additional_fields():
    return {'input_lengths', 'input_probs'}


@dataclass
class Settings:
    # can use an extension to render to file, e.g. 'png', or 'show' to plot
    visualize_mode: str = None

    # which data to load
    task_data_keys: Optional[Sequence[str]] = (DataKeys.ucl,)

    # task specific settings, see TaskSettings
    task_settings: MutableMapping[str, TaskSettings] = field(default_factory=_default_task_settings)

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

    def get_data_preprocessors(self):
        result = dict()
        for key in self.task_settings:
            if self.task_settings[key].preprocessor is not None:
                result[key] = self.task_settings[key].preprocessor
        if len(result) == 0:
            return None
        return result
