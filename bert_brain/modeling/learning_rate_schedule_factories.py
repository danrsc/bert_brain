import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from torch.optim.lr_scheduler import LambdaLR

# all schedules other than linear_warmup_rsqrt_decay are copied from
# https://github.com/huggingface/transformers/ \
# blob/dc17f2a1110aed8d1729e77b0619601e3d96b84e/src/transformers/optimization.py
# We copy these here so we can have different num_training_steps for different parts of the model


__all__ = [
    'LearningRateScheduleFactory',
    'ConstantLearningRateScheduleFactory',
    'ConstantWithWarmupLearningRateScheduleFactory',
    'LinearWithWarmupLearningRateScheduleFactory',
    'CosineWithWarmupLearningRateScheduleFactory',
    'CosineWithHardRestartsWithWarmupLearningRateScheduleFactory',
    'LinearWarmupSqrtDecayLearningRateScheduleFactory']


class LearningRateScheduleFactory:

    def get_schedule(self, optimizer, optimizer_grouped_parameters, last_epoch=-1):
        functions = list()
        for group in optimizer_grouped_parameters:
            functions.append(self._get_schedule_fn(group['t_total']))
        return LambdaLR(optimizer, functions, last_epoch)

    def _get_schedule_fn(self, num_training_steps):
        raise NotImplementedError('{} does not implement _get_schedule_fn'.format(type(self)))

    @classmethod
    def _handle_proportion(cls, maybe_proportion: Union[float, int], steps: int):
        if isinstance(maybe_proportion, float) and 0 < maybe_proportion <= 1.0:
            return int(np.floor(maybe_proportion * steps))
        return maybe_proportion


class ConstantLearningRateScheduleFactory(LearningRateScheduleFactory):

    def _get_schedule_fn(self, num_training_steps):
        def lr_lambda(_):
            return 1.0
        return lr_lambda


@dataclass(frozen=True)
class ConstantWithWarmupLearningRateScheduleFactory(LearningRateScheduleFactory):
    num_warmup_steps: Union[int, float] = 0.1

    def _get_schedule_fn(self, num_training_steps):
        num_warmup_steps = type(self)._handle_proportion(self.num_warmup_steps, num_training_steps)

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1.0, num_warmup_steps))
            return 1.0

        return lr_lambda


@dataclass(frozen=True)
class LinearWithWarmupLearningRateScheduleFactory(LearningRateScheduleFactory):
    num_warmup_steps: Union[int, float] = 0.1

    def _get_schedule_fn(self, num_training_steps):
        num_warmup_steps = type(self)._handle_proportion(self.num_warmup_steps, num_training_steps)

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

        return lr_lambda


@dataclass(frozen=True)
class CosineWithWarmupLearningRateScheduleFactory(LearningRateScheduleFactory):
    num_warmup_steps: Union[int, float] = 0.1
    num_cycles: int = 1

    def _get_schedule_fn(self, num_training_steps):
        num_warmup_steps = type(self)._handle_proportion(self.num_warmup_steps, num_training_steps)

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)))

        return lr_lambda


@dataclass(frozen=True)
class CosineWithHardRestartsWithWarmupLearningRateScheduleFactory(LearningRateScheduleFactory):
    num_warmup_steps: Union[int, float] = 0.1
    num_cycles: int = 1

    def _get_schedule_fn(self, num_training_steps):
        num_warmup_steps = type(self)._handle_proportion(self.num_warmup_steps, num_training_steps)

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            if progress >= 1.0:
                return 0.0
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(self.num_cycles) * progress) % 1.0))))

        return lr_lambda


@dataclass(frozen=True)
class LinearWarmupSqrtDecayLearningRateScheduleFactory(LearningRateScheduleFactory):
    num_warmup_steps: Optional[Union[int, float]] = 0.1

    def _get_schedule_fn(self, num_training_steps):
        num_warmup_steps = type(self)._handle_proportion(self.num_warmup_steps, num_training_steps)

        def lr_lambda(current_step: int) -> float:
            if current_step == 0:
                return 0
            return min(current_step ** -0.5, current_step * num_warmup_steps ** -1.5) / num_warmup_steps ** -0.5

        return lr_lambda
