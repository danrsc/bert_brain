from dataclasses import dataclass
import dataclasses
from typing import Optional, Any
from collections import OrderedDict

import numpy as np
import torch


__all__ = [
    'NoValidInputs',
    'logical_not',
    'masked_squared_error',
    'masked_pearsons_distance',
    'masked_cross_entropy',
    'masked_binary_cross_entropy_with_logits',
    'masked_soft_label_cross_entropy',
    'stop_word_and_target_not_nan_mask',
    'NamedTargetStopWordAwareMSE',
    'NamedTargetStopWordAwarePearsonDistance',
    'NamedTargetStopWordAwareBinaryCrossEntropyWithLogits',
    'NamedTargetStopWordAwareCrossEntropy',
    'NamedTargetStopWordAwareSoftLabelCrossEntropy',
    'NamedTargetSingleBinaryCrossEntropyWithLogits',
    'NamedTargetSingleCrossEntropy',
    'NamedTargetSingleSoftLabelCrossEntropy',
    'CriticMapping',
    'CriticKeys',
    'make_loss_handler']


class NoValidInputs(Exception):

    def __init__(self):
        super().__init__()


def logical_not(t):
    # use xor with 1 to give a logical not
    return t ^ 1


def masked_squared_error(mask, predictions, target):
    valid_count = mask.sum().item()
    if valid_count == 0:
        raise NoValidInputs()
    sq_err = (torch.masked_select(predictions, mask) - torch.masked_select(target, mask)) ** 2
    result = torch.zeros_like(target)
    result.masked_scatter_(mask, sq_err)
    return result, valid_count


def _values_or_zeros(mask, source):
    # this seems inefficient, but other ways I've tried mess up the gradient
    result = torch.zeros_like(source)
    result.masked_scatter_(mask, torch.masked_select(source, mask))
    return result


def masked_pearsons_distance(mask, predictions, target, sequence_axis=1):
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
    predictions = _values_or_zeros(mask, predictions)
    target = _values_or_zeros(mask, target)

    # compute the mean
    mean_input = predictions.sum(dim=sequence_axis, keepdim=True) / valid_counts_per_example
    mean_target = predictions.sum(dim=sequence_axis, keepdim=True) / valid_counts_per_example

    # remove the mean, and re-mask
    predictions = _values_or_zeros(mask, predictions - mean_input)
    target = _values_or_zeros(mask, target - mean_target)

    # compute the variance
    var_input = (predictions ** 2).sum(dim=sequence_axis, keepdim=True) / (valid_counts_per_example - 1)
    var_target = (target ** 2).sum(dim=sequence_axis, keepdim=True) / (valid_counts_per_example - 1)

    # min_value is an epsilon to avoid divide-by-zero, and to prevent sqrt from blowing up numerically
    min_value = torch.zeros((), dtype=var_input.dtype, device=var_input.device) + 1e-8
    var_input = torch.max(var_input, min_value)
    var_target = torch.max(var_target, min_value)

    # scale by the std
    predictions = predictions / torch.sqrt(var_input)
    target = target / torch.sqrt(var_target)

    # now r is straightforward to compute
    r = (predictions * target).sum(dim=sequence_axis, keepdim=True) / (valid_counts_per_example - 1)

    # convert to distance
    distance = 1 - r

    return distance, valid_count, var_input, var_target, mean_input, mean_target


def masked_cross_entropy(mask, predictions, target):

    valid_count = mask.sum().item()
    if valid_count == 0:
        raise NoValidInputs()
    predictions = predictions.view(np.prod(predictions.size()[:-1]), predictions.size()[-1])
    target = target.view(-1)
    flat_mask = mask.view(-1)
    valid_indices = torch.nonzero(flat_mask)
    predictions = predictions[valid_indices]
    target = target[valid_indices]
    loss = torch.nn.functional.cross_entropy(predictions, target, reduction='none')
    result = torch.zeros(mask.size(), dtype=loss.dtype, device=loss.device)
    result.masked_scatter_(mask, loss)
    return result, valid_count


def masked_binary_cross_entropy_with_logits(mask, predictions, target, pos_weight=None):
    valid_count = mask.sum().item()
    if valid_count == 0:
        raise NoValidInputs()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.masked_select(predictions, mask),
        torch.masked_select(target, mask),
        reduction='none',
        pos_weight=pos_weight)
    result = torch.zeros(mask.size(), dtype=loss.dtype, device=loss.device)
    result.masked_scatter_(mask, loss)
    return result, valid_count


def masked_soft_label_cross_entropy(mask, predictions, target):
    # note we just assume that the target values sum to 1 along axis=-1
    if mask is not None:
        valid_count = mask.sum().item()
        if valid_count == 0:
            raise NoValidInputs()
    else:
        valid_count = None

    # set up 1s in the prediction where the mask is False;
    # this will mean that log_softmax does not give an nan in case the predictions are
    # strange where they are meaningless
    if mask is not None:
        safer_input = torch.ones_like(predictions)
        safer_input.masked_scatter_(mask.view(mask.size() + (1,)), predictions)
    else:
        safer_input = predictions

    softmax = torch.nn.functional.log_softmax(safer_input, dim=-1)
    terms = -softmax * target
    cross_entropy = terms.sum(dim=-1)
    if mask is not None:
        cross_entropy = _values_or_zeros(mask, cross_entropy)
    else:
        valid_count = np.prod(cross_entropy.size())
    return cross_entropy, valid_count


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
    mask: Optional[np.array]
    prediction: np.array
    target: np.array
    sequence_type: str
    data_set_id: Optional[int] = None
    unique_id: Optional[int] = None


def _masked_reduce(loss, valid_count, reduction, as_numpy):
    if as_numpy:
        loss = loss.detach().cpu().numpy()
    if reduction == 'mean' or reduction == 'sum':
        loss = loss.sum()
        if as_numpy:
            loss = loss.item()
        if reduction == 'mean':
            return loss / valid_count
        return loss
    if reduction != 'none':
        raise ValueError('Unknown value for reduction: {}'.format(reduction))
    return loss, valid_count


class _NamedTargetMaskedLoss:

    def __init__(self, field, weight=1.):
        self.field, self.weight = field, weight

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

    def __call__(
            self, batch, prediction_dict, return_detailed=False, reduction='mean', as_numpy=False, apply_weight=True):
        predictions = prediction_dict[self.field]
        target = batch[self.field]
        mask = self._get_mask(batch, predictions, target)

        try:
            result, valid_count = self._masked_loss(mask, predictions, target)
            result = _masked_reduce(result, valid_count, reduction, as_numpy)
        except NoValidInputs:
            if reduction == 'mean' or reduction == 'sum':
                result = 'no_valid_inputs'
            else:
                result = 'no_valid_inputs', 0

        if apply_weight:
            result = self.apply_weight(result)

        if return_detailed:

            example_indices = None
            group_prediction_key = (self.field, 'example_ids')
            if group_prediction_key in prediction_dict:
                example_indices = prediction_dict[group_prediction_key].detach().cpu().numpy()

            batch_mask = mask.detach().cpu().numpy()
            batch_predictions = predictions.detach().cpu().numpy()
            batch_target = target.detach().cpu().numpy()
            batch_mask = np.split(batch_mask, len(batch_mask)) if len(batch_mask) > 0 else batch_mask
            batch_predictions = np.split(batch_predictions, len(batch_predictions)) \
                if len(batch_predictions) > 0 else batch_predictions
            batch_target = np.split(batch_target, len(batch_target)) if len(batch_target) > 0 else batch_target

            sequence_type = 'sequence' if self._is_sequence_loss() else 'single'

            if example_indices is not None:  # group by the example indices
                sequence_type = 'grouped'
                grouped = dict()
                for m, p, t, ex in zip(batch_mask, batch_predictions, batch_target, example_indices):
                    if ex not in grouped:
                        grouped[ex] = (list(), list(), list())
                    grouped[ex][0].append(np.expand_dims(m, 1))
                    grouped[ex][1].append(np.expand_dims(p, 1))
                    grouped[ex][2].append(np.expand_dims(t, 1))
                batch_mask = list()
                batch_predictions = list()
                batch_target = list()
                example_indices = [ex for ex in sorted(grouped)]
                for ex in example_indices:
                    batch_mask.append(np.concatenate(grouped[ex][0], axis=1))
                    batch_predictions.append(np.concatenate(grouped[ex][1], axis=1))
                    batch_target.append(np.concatenate(grouped[ex][2], axis=1))

            detailed_result = list()
            for idx, (example_mask, example_predictions, example_targets) in enumerate(zip(
                    batch_mask, batch_predictions, batch_target)):

                if example_indices is not None:
                    idx = example_indices[idx]

                data_set_id = batch['data_set_id'][idx] if 'data_set_id' in batch else None
                unique_id = batch['unique_id'][idx] if 'unique_id' in batch else None
                detailed_result.append(
                    DetailedResult(
                        mask=np.squeeze(example_mask, axis=0) == 1,  # convert to bool
                        prediction=np.squeeze(example_predictions, axis=0),
                        target=np.squeeze(example_targets, axis=0),
                        sequence_type=sequence_type,
                        data_set_id=data_set_id,
                        unique_id=unique_id))
            return result, detailed_result
        return result

    def _get_mask(self, batch, predictions, target):
        return logical_not(torch.isnan(target))

    def _masked_loss(self, mask, predictions, target):
        raise NotImplementedError('{} does not implement _masked_loss'.format(type(self)))

    @classmethod
    def _is_sequence_loss(cls):
        return False

    def shape_adjust(self, shape):
        return shape


class _NamedTargetStopWordAwareLoss(_NamedTargetMaskedLoss):

    def __init__(self, field, keep_content=True, weight=1.):
        super().__init__(field, weight)
        self.keep_content = keep_content

    def _get_mask(self, batch, predictions, target):
        return stop_word_and_target_not_nan_mask(
            self.keep_content, target, batch['is_stop'], batch['is_begin_word_pieces'])

    def _masked_loss(self, mask, predictions, target):
        raise NotImplementedError('{} does not implement _masked_loss'.format(type(self)))

    @classmethod
    def _is_sequence_loss(cls):
        return True


class NamedTargetStopWordAwareMSE(_NamedTargetStopWordAwareLoss):

    def _masked_loss(self, mask, predictions, target):
        return masked_squared_error(mask, predictions, target)


class NamedTargetStopWordAwarePearsonDistance(_NamedTargetStopWordAwareLoss):

    def __init__(self, field, keep_content=True, should_penalize_scale=False, weight=1.):
        super().__init__(field, keep_content, weight)
        self.should_penalize_scale = should_penalize_scale

    def _masked_loss(self, mask, predictions, target):
        distance, valid_count, var_input, var_target, mean_input, mean_target = masked_pearsons_distance(
            mask, predictions, target)
        loss = distance
        if self.should_penalize_scale:
            loss = loss + (var_input - var_target) ** 2
        return loss, valid_count


class NamedTargetStopWordAwareCrossEntropy(_NamedTargetStopWordAwareLoss):

    def __init__(self, field, num_classes, keep_content=True, weight=1.):
        self.num_classes = num_classes
        super().__init__(field, keep_content, weight)

    def _masked_loss(self, mask, predictions, target):
        return masked_cross_entropy(mask, predictions, target)

    def shape_adjust(self, shape):
        return shape + (self.num_classes,)


class NamedTargetStopWordAwareBinaryCrossEntropyWithLogits(_NamedTargetStopWordAwareLoss):

    def __init__(self, field, keep_content=True, weight=1., pos_weight=None):
        super().__init__(field, keep_content, weight)
        self.pos_weight = pos_weight

    def _masked_loss(self, mask, predictions, target):
        return masked_binary_cross_entropy_with_logits(mask, predictions, target, self.pos_weight)


class NamedTargetStopWordAwareSoftLabelCrossEntropy(_NamedTargetStopWordAwareLoss):

    def _masked_loss(self, mask, predictions, target):
        return masked_soft_label_cross_entropy(mask, predictions, target)


class NamedTargetSingleMSE(_NamedTargetMaskedLoss):

    def _masked_loss(self, mask, predictions, target):
        return masked_squared_error(mask, predictions, target)


class NamedTargetSingleCrossEntropy(_NamedTargetMaskedLoss):

    def __init__(self, field, num_classes, weight=1.):
        self.num_classes = num_classes
        super().__init__(field, weight)

    def _masked_loss(self, mask, predictions, target):
        return masked_cross_entropy(mask, predictions, target)

    def shape_adjust(self, shape):
        return shape + (self.num_classes,)


class NamedTargetSingleSoftLabelCrossEntropy(_NamedTargetMaskedLoss):

    def _masked_loss(self, mask, predictions, target):
        return masked_soft_label_cross_entropy(mask, predictions, target)


class NamedTargetSingleBinaryCrossEntropyWithLogits(_NamedTargetMaskedLoss):

    def __init__(self, field, weight=1., pos_weight=None):
        super().__init__(field, weight)
        self.pos_weight = pos_weight

    def _masked_loss(self, mask, predictions, target):
        return masked_binary_cross_entropy_with_logits(mask, predictions, target, self.pos_weight)


@dataclass(frozen=True)
class CriticMapping:
    # this metadata trick allows us to give the canonical value along with the field definition
    # while not specifying a default (so we force all versions of the mapping to instantiate all the fields)
    mse: Any = dataclasses.field(metadata=dict(hidden_value=NamedTargetStopWordAwareMSE))
    pearson: Any = dataclasses.field(metadata=dict(hidden_value=NamedTargetStopWordAwarePearsonDistance))
    cross_entropy: Any = dataclasses.field(metadata=dict(hidden_value=NamedTargetStopWordAwareCrossEntropy))
    binary_cross_entropy: Any = dataclasses.field(
        metadata=dict(hidden_value=NamedTargetStopWordAwareBinaryCrossEntropyWithLogits))
    soft_label_cross_entropy: Any = dataclasses.field(
        metadata=dict(hidden_value=NamedTargetStopWordAwareSoftLabelCrossEntropy))
    single_mse: Any = dataclasses.field(metadata=dict(hidden_value=NamedTargetSingleMSE))
    single_cross_entropy: Any = dataclasses.field(
        metadata=dict(hidden_value=NamedTargetSingleCrossEntropy))
    single_binary_cross_entropy: Any = dataclasses.field(
        metadata=dict(hidden_value=NamedTargetSingleBinaryCrossEntropyWithLogits))
    single_soft_label_cross_entropy: Any = dataclasses.field(
        metadata=dict(hidden_value=NamedTargetSingleSoftLabelCrossEntropy))


CriticKeys = CriticMapping(**dict((f.name, f.name) for f in dataclasses.fields(CriticMapping)))
_critic_type_dict = OrderedDict((f.name, f.metadata['hidden_value']) for f in dataclasses.fields(CriticMapping))


def make_loss_handler(field, which_loss, loss_kwargs=None):
    if which_loss not in _critic_type_dict:
        raise ValueError('Unknown value for which_loss. Known values are: {}'.format(_critic_type_dict.keys()))
    factory = _critic_type_dict[which_loss]
    if loss_kwargs is None:
        loss_kwargs = {}
    return factory(field, **loss_kwargs)
