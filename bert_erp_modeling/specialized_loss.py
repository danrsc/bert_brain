from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


__all__ = ['NoValidInputs', 'logical_not', 'masked_squared_error', 'stop_word_and_target_not_nan_mask',
           'NamedTargetStopWordMSE']


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


def stop_word_and_target_not_nan_mask(keep_content, target, is_stop, is_begin_word_pieces):
    if is_stop is not None:
        if len(is_stop.size()) < len(target.size()):
            is_stop = is_stop.view(is_stop.size() + (1,) * (len(target.size()) - len(is_stop.size())))
        is_keep = logical_not(is_stop) if keep_content else is_stop
        if is_begin_word_pieces is not None:
            return is_keep & logical_not(torch.isnan(target)) & is_begin_word_pieces
        else:
            return is_keep & logical_not(torch.isnan(target))
    else:
        if is_begin_word_pieces is not None:
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


class NamedTargetStopWordMSE:

    def __init__(self, field, keep_content, weight=1.):
        self.field = field
        self.weight = weight
        self.keep_content = keep_content

    def __call__(self, batch, predictions, return_detailed=False):
        predictions = predictions[self.field]
        target = batch[self.field]
        mask = stop_word_and_target_not_nan_mask(
            self.keep_content, target, batch['input_is_stop'], batch['input_is_begin_word_pieces'])
        sq_err, valid_count = masked_squared_error(mask, predictions, target)
        result = sq_err, valid_count
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
