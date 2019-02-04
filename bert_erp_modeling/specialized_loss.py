from collections import OrderedDict
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
    unique_id: Optional[int] = None


class NamedTargetStopWordMSE:

    def __init__(self, keep_content, ordered_shape_dict):
        self.keep_content = keep_content
        self.shapes = OrderedDict(ordered_shape_dict)
        self.splits = list()
        for k in self.shapes:
            self.splits.append(np.prod(self.shapes[k]))

    def __call__(self, predictions, target, is_stop, is_begin_word_pieces, return_detailed=False, unique_ids=None):
        split_predictions = torch.split(predictions, self.splits, dim=-1)
        split_target = torch.split(target, self.splits, dim=-1)
        result = OrderedDict()
        detailed_result = OrderedDict() if return_detailed else None
        for k, k_predictions, k_target in zip(self.shapes, split_predictions, split_target):
            k_predictions = k_predictions.view(k_predictions.size()[0] + self.shapes[k])
            k_target = k_target.view(k_target.size()[0] + self.shapes[k])
            mask = stop_word_and_target_not_nan_mask(self.keep_content, k_target, is_stop, is_begin_word_pieces)
            sq_err, valid_count = masked_squared_error(mask, k_predictions, k_target)
            result[k] = sq_err, valid_count
            if return_detailed:
                batch_mask = mask.detach.cpu.numpy()
                batch_predictions = k_predictions.detach.cpu().numpy()
                batch_target = k_target.detach.cpu().numpy()
                batch_mask = np.split(batch_mask, len(batch_mask))
                batch_predictions = np.split(batch_predictions, len(batch_predictions))
                batch_target = np.split(batch_target, len(batch_target))
                detailed_result[k] = list()
                for idx, (example_mask, example_predictions, example_targets) in enumerate(zip(
                        batch_mask, batch_predictions, batch_target)):
                    unique_id = unique_ids[idx].item() if unique_ids is not None else None
                    detailed_result[k].append(
                        DetailedResult(example_mask, example_predictions, example_targets, unique_id))
        if return_detailed:
            return result, detailed_result
        return result
