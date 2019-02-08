from dataclasses import dataclass, field
from typing import Any, Sequence, Callable, MutableMapping, Optional

import numpy as np
from bert_erp_datasets import DataLoader, PreprocessMany, PreprocessStandardize, PreprocessLog, PreprocessPCA, \
    PreprocessMakeBinary, PreprocessNanMean


__all__ = ['TaskSettings', 'Settings']


@dataclass
class TaskSettings:
    critic_type: str
    fill_value: Any
    preprocessor: Optional[Callable] = None
    # stop_word_mode can be:
    #     None: no distinction made between content words and stop words
    #     'train_both': training is done on both, results are reported separately
    #     'train_content': training is done on content, both results reported
    #     'report_content': training is done on content, only content results reported
    stop_word_mode: Optional[str] = 'train_content'


def _default_task_settings():

    # take boxcox transform of these
    ucl_log_keys = {'first_fixation', 'first_pass', 'right_bounded', 'go_past', 'reading_time'}

    preprocess_standardize = PreprocessStandardize(stop_mode='content')

    return {
        DataLoader.ucl: TaskSettings(
            critic_type='mse',
            fill_value=np.nan,
            preprocessor=PreprocessMany(
                PreprocessLog(data_key_whitelist=ucl_log_keys),
                preprocess_standardize)),
    }


@dataclass
class Settings:
    # can use an extension to render to file, e.g. 'png', or 'show' to plot
    visualize_mode: str = None

    # which sets to load for the decoder step
    task_data_keys: Optional[Sequence[str]] = (DataLoader.ucl,)

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

    loss_tasks: set = field(default_factory=set)
    loss_data: set = field(default_factory=set)

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
