from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


__all__ = ['NoValidInputs', 'logical_not', 'masked_squared_error', 'stop_word_and_target_not_nan_mask',
           'NamedTargetStopWordAwareMSE', 'NamedTargetStopWordAwarePearsonDistance']


class NoValidInputs(Exception):

    def __init__(self):
        super().__init__()


def logical_not(t):
    # use xor with 1 to give a logical not
    return t ^ 1


def masked_squared_error(mask, input, target):
    valid_count = mask.sum().item()
    if valid_count == 0:
        raise NoValidInputs()
    sq_err = (torch.masked_select(input, mask) - torch.masked_select(target, mask)) ** 2
    result = torch.zeros_like(target)
    result.masked_scatter_(mask, sq_err)
    return result, valid_count


def _values_or_zeros(mask, source):
    # this seems inefficient, but other ways I've tried mess up the gradient
    result = torch.zeros_like(source)
    result.masked_scatter_(mask, torch.masked_select(source, mask))
    return result


def masked_pearsons_distance(mask, input, target, sequence_axis=1):
    valid_counts_per_example = mask.sum(dim=sequence_axis, keepdim=True)
    # wherever valid_counts_per_example is less than 2, we need to set the mask to False
    indicator_valid_example = valid_counts_per_example > 1
    mask = mask & indicator_valid_example
    valid_counts_per_example = mask.sum(dim=sequence_axis, keepdim=True)

    # ignore the values where valid_counts_per_example == 0, distance will already be 0 at these locations
    valid_count = (valid_counts_per_example > 0).sum().item()

    if valid_count == 0:
        raise NoValidInputs()

    # this way of computing is more numerically stable then some alternatives

    # replace masked values with zero
    input = _values_or_zeros(mask, input)
    target = _values_or_zeros(mask, target)

    # compute the mean
    mean_input = input.sum(dim=sequence_axis, keepdim=True) / valid_counts_per_example
    mean_target = input.sum(dim=sequence_axis, keepdim=True) / valid_counts_per_example

    # remove the mean, and re-mask
    input = _values_or_zeros(mask, input - mean_input)
    target = _values_or_zeros(mask, target - mean_target)

    # compute the variance
    var_input = (input ** 2).sum(dim=sequence_axis, keepdim=True) / (valid_counts_per_example - 1)
    var_target = (target ** 2).sum(dim=sequence_axis, keepdim=True) / (valid_counts_per_example - 1)

    # min_value is an epsilon to avoid divide-by-zero, and to prevent sqrt from blowing up numerically
    min_value = torch.zeros((), dtype=var_input.dtype, device=var_input.device) + 1e-8
    var_input = torch.max(var_input, min_value)
    var_target = torch.max(var_target, min_value)

    # scale by the std
    input = input / torch.sqrt(var_input)
    target = target / torch.sqrt(var_target)

    # now r is straightforward to compute
    r = (input * target).sum(dim=sequence_axis, keepdim=True) / (valid_counts_per_example - 1)

    # convert to distance
    distance = 1 - r

    return distance, valid_count, var_input, var_target, mean_input, mean_target


def stop_word_and_target_not_nan_mask(keep_content, target, is_stop, is_begin_word_pieces):
    if is_stop is not None:
        if len(is_stop.size()) < len(target.size()):
            is_stop = is_stop.view(is_stop.size() + (1,) * (len(target.size()) - len(is_stop.size())))
        is_keep = logical_not(is_stop) if keep_content else is_stop
        if is_begin_word_pieces is not None:
            if len(is_begin_word_pieces.size()) < len(target.size()):
                is_begin_word_pieces = is_begin_word_pieces.view(
                    is_begin_word_pieces.size() + (1,) * (len(target.size()) - len(is_begin_word_pieces.size())))
            return is_keep & logical_not(torch.isnan(target)) & is_begin_word_pieces
        else:
            return is_keep & logical_not(torch.isnan(target))
    else:
        if is_begin_word_pieces is not None:
            if len(is_begin_word_pieces.size()) < len(target.size()):
                is_begin_word_pieces = is_begin_word_pieces.view(
                    is_begin_word_pieces.size() + (1,) * (len(target.size()) - len(is_begin_word_pieces.size())))
            return logical_not(torch.isnan(target)) & is_begin_word_pieces
        else:
            return logical_not(torch.isnan(target))


@dataclass
class DetailedResult:
    mask: np.array
    prediction: np.array
    target: np.array
    data_set_id: Optional[int] = None
    unique_id: Optional[int] = None


class _NamedTargetStopWordAwareLoss:

    def __init__(self, field, keep_content, weight):
        self.field, self.keep_content, self.weight = field, keep_content, weight

    def apply_weight(self, result):
        is_tuple = isinstance(result, tuple)
        if is_tuple:
            loss = result[0]
        else:
            loss = result
        if isinstance(loss, str):
            assert(loss == 'no_valid_inputs')
        loss = self.weight * loss
        if is_tuple:
            return (loss,) + result[1:]
        return loss

    def __call__(self, batch, predictions, return_detailed=False, reduction='mean', as_numpy=False, apply_weight=True):
        predictions = predictions[self.field]
        target = batch[self.field]
        mask = stop_word_and_target_not_nan_mask(
            self.keep_content, target, batch['input_is_stop'], batch['input_is_begin_word_pieces'])
        result = self._masked_loss(mask, predictions, target, reduction, as_numpy)
        if apply_weight:
            result = self.apply_weight(result)

        if return_detailed:
            batch_mask = mask.detach().cpu().numpy()
            batch_predictions = predictions.detach().cpu().numpy()
            batch_target = target.detach().cpu().numpy()
            batch_mask = np.split(batch_mask, len(batch_mask))
            batch_predictions = np.split(batch_predictions, len(batch_predictions))
            batch_target = np.split(batch_target, len(batch_target))
            detailed_result = list()
            for idx, (example_mask, example_predictions, example_targets) in enumerate(zip(
                    batch_mask, batch_predictions, batch_target)):
                data_set_id = batch['data_set_id'][idx] if 'data_set_id' in batch else None
                unique_id = batch['unique_id'][idx] if 'unique_id' in batch else None
                detailed_result.append(
                    DetailedResult(
                        np.squeeze(example_mask, axis=0) == 1,  # convert to bool
                        np.squeeze(example_predictions, axis=0),
                        np.squeeze(example_targets, axis=0),
                        data_set_id,
                        unique_id))
            return result, detailed_result
        return result

    def _masked_loss(self, mask, predictions, target, reduction, as_numpy):
        raise NotImplementedError('{} does not implement _masked_loss'.format(type(self)))


class NamedTargetStopWordAwareMSE(_NamedTargetStopWordAwareLoss):

    def _masked_loss(self, mask, predictions, target, reduction, as_numpy):
        try:
            sq_err, valid_count = masked_squared_error(mask, predictions, target)
            if as_numpy:
                sq_err = sq_err.detach().cpu().numpy()
                if reduction == 'mean' or reduction == 'sum':
                    sq_err = sq_err.sum().item()
                    if reduction == 'mean':
                        return sq_err / valid_count
                    return sq_err
                return sq_err, valid_count
            if reduction == 'mean' or reduction == 'sum':
                sq_err = sq_err.sum()
                if reduction == 'mean':
                    return sq_err / valid_count
                return sq_err
            return sq_err, valid_count
        except NoValidInputs:
            if reduction == 'mean' or reduction == 'sum':
                return 'no_valid_inputs'
            return 'no_valid_inputs', 0


class NamedTargetStopWordAwarePearsonDistance(_NamedTargetStopWordAwareLoss):

    def __init__(self, field, keep_content, should_penalize_scale=False, weight=1.):
        super().__init__(field, keep_content, weight)
        self.should_penalize_scale = should_penalize_scale

    def _masked_loss(self, mask, predictions, target, reduction, as_numpy):
        try:
            distance, valid_count, var_input, var_target, mean_input, mean_target = masked_pearsons_distance(
                mask, predictions, target)
            loss = distance
            if self.should_penalize_scale:
                loss = loss + (var_input - var_target) ** 2
            if as_numpy:
                loss = loss.detach().cpu().numpy()
            if reduction == 'mean' or reduction == 'sum':
                loss = loss.sum()
                if as_numpy:
                    loss = loss.item()
                if reduction == 'mean':
                    return loss / valid_count
                return loss
            return loss, valid_count
        except NoValidInputs:
            if reduction == 'mean' or reduction == 'sum':
                return 'no_valid_inputs'
            return 'no_valid_inputs', 0
