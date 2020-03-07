from dataclasses import dataclass
import dataclasses
from typing import Optional, Callable, List

import numpy as np
import torch
import torch.nn
import torch.nn.functional


__all__ = [
    'NoValidInputs',
    'masked_squared_error',
    'masked_absolute_error',
    'masked_pearsons_distance',
    'masked_cross_entropy',
    'masked_negative_log_likelihood',
    'masked_binary_cross_entropy_with_logits',
    'masked_binary_cross_entropy',
    'masked_soft_label_cross_entropy',
    'masked_soft_label_negative_log_likelihood',
    'stop_word_and_target_not_nan_mask',
    'NamedTargetMaskedLossBase',
    'NamedTargetStopWordAwareMSE',
    'NamedTargetStopWordAwareMAE',
    'NamedTargetStopWordAwareKLeastSE',
    'NamedTargetStopWordAwareKLeastSEEvalUpdate',
    'NamedTargetStopWordAwareKLeastAE',
    'NamedTargetStopWordAwareKLeastAEEvalUpdate',
    'NamedTargetStopWordAwarePearsonDistance',
    'NamedTargetStopWordAwareBinaryCrossEntropyWithLogits',
    'NamedTargetStopWordAwareBinaryCrossEntropy',
    'NamedTargetStopWordAwareCrossEntropy',
    'NamedTargetStopWordAwareNLL',
    'NamedTargetStopWordAwareSoftLabelCrossEntropy',
    'NamedTargetStopWordAwareSoftLabelNLL',
    'NamedTargetSingleMSE',
    'NamedTargetSingleMAE',
    'NamedTargetSingleKLeastSE',
    'NamedTargetSingleKLeastSEEvalUpdate',
    'NamedTargetSingleKLeastAE',
    'NamedTargetSingleKLeastAEEvalUpdate',
    'NamedTargetSinglePearsonDistance',
    'NamedTargetSingleBinaryCrossEntropyWithLogits',
    'NamedTargetSingleBinaryCrossEntropy',
    'NamedTargetSingleCrossEntropy',
    'NamedTargetSingleNLL',
    'NamedTargetSingleSoftLabelCrossEntropy',
    'NamedTargetSingleSoftLabelNLL',
    'KLeastSEHalvingEpochs',
    'DetailedResult']


class NoValidInputs(Exception):
    def __init__(self):
        super().__init__()


def masked_squared_error(mask, predictions, target):
    valid_count = mask.sum().item()
    if valid_count == 0:
        raise NoValidInputs()
    sq_err = (torch.masked_select(predictions, mask) - torch.masked_select(target, mask)) ** 2
    result = torch.zeros_like(target)
    result.masked_scatter_(mask, sq_err)
    return result, valid_count


def masked_absolute_error(mask, predictions, target):
    valid_count = mask.sum().item()
    if valid_count == 0:
        raise NoValidInputs()
    abs_err = torch.abs(torch.masked_select(predictions, mask) - torch.masked_select(target, mask))
    result = torch.zeros_like(target)
    result.masked_scatter_(mask, abs_err)
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

    # convert type on counts for computations
    valid_counts_per_example = valid_counts_per_example.type(target.dtype)

    # this way of computing is more numerically stable than some alternatives

    # replace masked values with zero
    predictions = _values_or_zeros(mask, predictions)
    target = _values_or_zeros(mask, target)

    # compute the mean
    mean_predictions = predictions.sum(dim=sequence_axis, keepdim=True) / valid_counts_per_example
    mean_target = target.sum(dim=sequence_axis, keepdim=True) / valid_counts_per_example

    # remove the mean, and re-mask
    predictions = _values_or_zeros(mask, predictions - mean_predictions)
    target = _values_or_zeros(mask, target - mean_target)

    # compute the variance
    var_predictions = (predictions ** 2).sum(dim=sequence_axis, keepdim=True) / (valid_counts_per_example - 1)
    var_target = (target ** 2).sum(dim=sequence_axis, keepdim=True) / (valid_counts_per_example - 1)

    # min_value is an epsilon to avoid divide-by-zero, and to prevent sqrt from blowing up numerically
    min_value = torch.zeros((), dtype=var_predictions.dtype, device=var_predictions.device) + 1e-8
    safe_var_predictions = torch.max(var_predictions, min_value)
    safe_var_target = torch.max(var_target, min_value)

    # scale by the std
    predictions = predictions / torch.sqrt(safe_var_predictions)
    target = target / torch.sqrt(safe_var_target)

    # now r is straightforward to compute
    r = (predictions * target).sum(dim=sequence_axis, keepdim=True) / (valid_counts_per_example - 1)

    # convert to distance
    distance = 1 - r

    # final masking to get rid of numerically unstable values
    distance = _values_or_zeros(indicator_valid_example, distance)

    return distance, valid_count, var_predictions, var_target, mean_predictions, mean_target


def masked_cross_entropy(mask, predictions, target):
    return _masked_cross_entropy(mask, predictions, target, use_nll=False)


def masked_negative_log_likelihood(mask, predictions, target):
    return _masked_cross_entropy(mask, predictions, target, use_nll=True)


def _masked_cross_entropy(mask, predictions, target, use_nll):
    valid_count = mask.sum().item()
    if valid_count == 0:
        raise NoValidInputs()
    predictions = predictions.reshape(np.prod(predictions.size()[:-1]), predictions.size()[-1])
    target = target.view(-1)
    flat_mask = mask.view(-1)
    valid_indices = torch.squeeze(torch.nonzero(flat_mask), dim=1)
    predictions = predictions[valid_indices]
    target = target[valid_indices].type(torch.long)
    if use_nll:
        loss = torch.nn.functional.nll_loss(predictions, target, reduction='none')
    else:
        loss = torch.nn.functional.cross_entropy(predictions, target, reduction='none')
    result = torch.zeros(mask.size(), dtype=loss.dtype, device=loss.device)
    result.masked_scatter_(mask, loss)
    return result, valid_count


def masked_binary_cross_entropy(mask, predictions, target, weight=None):
    valid_count = mask.sum().item()
    if valid_count == 0:
        raise NoValidInputs()
    loss = torch.nn.functional.binary_cross_entropy(
        torch.masked_select(predictions, mask),
        torch.masked_select(target, mask),
        reduction='none',
        weight=weight)
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


def masked_soft_label_negative_log_likelihood(mask, predictions, target):
    # note we just assume that the target values sum to 1 along axis=-1
    if mask is not None:
        valid_count = mask.sum().item()
        if valid_count == 0:
            raise NoValidInputs()
    else:
        valid_count = None

    # set up -20 in the prediction where the mask is False;
    if mask is not None:
        softmax = -20 * torch.ones_like(predictions)
        softmax.masked_scatter_(mask.view(mask.size() + (1,)), predictions)
    else:
        softmax = predictions

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
        is_keep = ~is_stop if keep_content else is_stop
        if is_begin_word_pieces is not None:
            if len(is_begin_word_pieces.size()) < len(target.size()):
                is_begin_word_pieces = is_begin_word_pieces.view(
                    is_begin_word_pieces.size() + (1,) * (len(target.size()) - len(is_begin_word_pieces.size())))
            return is_keep & ~torch.isnan(target) & is_begin_word_pieces
        else:
            return is_keep & ~torch.isnan(target)
    else:
        if is_begin_word_pieces is not None:
            if len(is_begin_word_pieces.size()) < len(target.size()):
                is_begin_word_pieces = is_begin_word_pieces.view(
                    is_begin_word_pieces.size() + (1,) * (len(target.size()) - len(is_begin_word_pieces.size())))
            return ~torch.isnan(target) & is_begin_word_pieces
        else:
            return ~torch.isnan(target)


def k_least_squared_error(
        is_eval, is_sequence, k, mask, predictions, target, accumulator, active_mask, moving_average_decay,
        use_abs=False):

    if is_sequence:
        flat_shape = (target.size()[0] * target.size()[1], int(np.prod(target.size()[2:])))
    else:
        flat_shape = (target.size()[0], int(np.prod(target.size()[1:])))

    flat_mask = torch.reshape(mask, flat_shape)

    if not is_eval:
        if use_abs:
            sq_err = torch.abs(torch.masked_select(predictions, mask) - torch.masked_select(target, mask))
        else:
            sq_err = (torch.masked_select(predictions, mask) - torch.masked_select(target, mask)) ** 2
        sq_err_or_zeros = torch.zeros_like(target)
        sq_err_or_zeros.masked_scatter_(mask, sq_err)
        sq_err_or_zeros = sq_err_or_zeros.detach()

        sq_err_or_zeros = torch.reshape(sq_err_or_zeros, flat_shape)

        flat_mask_float = flat_mask.type(sq_err_or_zeros.dtype)
        indices_terms = (torch.cumsum(flat_mask.type(torch.long), dim=0) - 1) * flat_mask.type(torch.long)
        num_terms = flat_mask.sum(dim=0, keepdim=True)

        # alpha^(num_terms - 1) * x_0 + \sum_i alpha^(num_terms - i - 1) (1 - alpha) x_i

        one_minus_alpha_coeff = (1 - moving_average_decay) * flat_mask_float

        if accumulator is None:
            assert(active_mask is None)
            first_terms = (indices_terms == 0) & flat_mask
            one_minus_alpha_coeff.masked_scatter_(first_terms, torch.ones_like(one_minus_alpha_coeff))
            indices_terms = indices_terms - 1
            num_terms = num_terms - 1
            alpha_coeff = torch.pow(
                moving_average_decay,
                num_terms.type(flat_mask_float.dtype) - indices_terms.type(flat_mask_float.dtype) - 1) * flat_mask_float
            alpha_coeff.masked_scatter_(first_terms, torch.ones_like(alpha_coeff))

            accumulator = torch.sum(alpha_coeff * one_minus_alpha_coeff * sq_err_or_zeros, dim=0, keepdim=True)
            active_mask = torch.sum(flat_mask, dim=0, keepdim=True) > 0
        else:
            alpha_coeff = torch.pow(
                moving_average_decay,
                num_terms.type(flat_mask_float.dtype) - indices_terms.type(flat_mask_float.dtype) - 1) * flat_mask_float

            accumulator = accumulator + torch.sum(
                alpha_coeff * one_minus_alpha_coeff * sq_err_or_zeros, dim=0, keepdim=True)
            current_active = torch.sum(flat_mask, dim=0, keepdim=True) > 0
            active_mask = active_mask | current_active
    elif accumulator is None:
        raise RuntimeError('Cannot call k_least_squared_error with is_eval=True and accumulator=None')

    if mask.size()[0] > 0:  # only do this if there are items in the batch for this loss
        # set scores to a value greater than everything in the accumulator
        scores = torch.zeros_like(accumulator) + accumulator.max() + 1
        # set scores to accumulator where it is valid
        scores.masked_scatter_(active_mask, accumulator)

        _, top_k = torch.topk(scores, k, len(scores.size()) - 1, largest=False, sorted=False)

        top_k_mask = torch.zeros(scores.size(), dtype=mask.dtype, device=mask.device)
        top_k_mask.scatter_(
            len(top_k_mask.size()) - 1, top_k, torch.ones(top_k.size(), dtype=mask.dtype, device=mask.device))

        top_k_mask = top_k_mask.repeat((flat_shape[0],) + (1,) * (len(top_k_mask.size()) - 1))
        top_k_mask = torch.reshape(top_k_mask, mask.size())
        mask = mask & top_k_mask

    err = masked_absolute_error(mask, predictions, target) \
        if use_abs else masked_squared_error(mask, predictions, target)
    return accumulator, active_mask, err


def update_k_least(accumulator, counts, k):
    safe_divisor = torch.max(torch.ones_like(counts), counts)
    mse = accumulator / safe_divisor.type(accumulator.dtype)
    # set scores to a value greater than everything in the accumulator
    scores = torch.zeros_like(mse) + mse.max() + 1
    # set scores to mse where it is valid
    scores.masked_scatter_(counts > 0, mse)

    _, top_k = torch.topk(scores, k, len(scores.size()) - 1, largest=False, sorted=False)

    top_k_mask = torch.zeros(scores.size(), dtype=torch.uint8, device=accumulator.device)
    top_k_mask.scatter_(
        len(top_k_mask.size()) - 1, top_k, torch.ones(top_k.size(), dtype=torch.uint8, device=accumulator.device))

    return top_k_mask


def k_least_squared_error_update_on_eval(
        is_eval, is_sequence, mask, predictions, target, top_k_mask, next_accumulator, next_counts, use_abs=False):

    if is_sequence:
        flat_shape = (target.size()[0] * target.size()[1], int(np.prod(target.size()[2:])))
    else:
        flat_shape = (target.size()[0], int(np.prod(target.size()[1:])))

    flat_mask = torch.reshape(mask, flat_shape)

    if is_eval:
        if use_abs:
            sq_err = torch.abs(torch.masked_select(predictions, mask) - torch.masked_select(target, mask))
        else:
            sq_err = (torch.masked_select(predictions, mask) - torch.masked_select(target, mask)) ** 2
        sq_err_or_zeros = torch.zeros_like(target)
        sq_err_or_zeros.masked_scatter_(mask, sq_err)
        sq_err_or_zeros = sq_err_or_zeros.detach()

        sq_err_or_zeros = torch.reshape(sq_err_or_zeros, flat_shape)

        if next_accumulator is None:
            assert(next_counts is None)
            next_accumulator = torch.sum(sq_err_or_zeros, dim=0, keepdim=True)
            next_counts = torch.sum(flat_mask, dim=0, keepdim=True)
        else:
            next_accumulator = next_accumulator + torch.sum(sq_err_or_zeros, dim=0, keepdim=True)
            next_counts = next_counts + torch.sum(flat_mask, dim=0, keepdim=True)

    if mask.size()[0] > 0 and top_k_mask is not None:
        top_k_mask = top_k_mask.repeat((flat_shape[0],) + (1,) * (len(top_k_mask.size()) - 1))
        top_k_mask = torch.reshape(top_k_mask, mask.size())
        mask = mask & top_k_mask

    err = masked_absolute_error(mask, predictions, target) \
        if use_abs else masked_squared_error(mask, predictions, target)
    return next_accumulator, next_counts, err


@dataclass
class DetailedResult:
    mask: Optional[np.ndarray]
    prediction: np.ndarray
    target: np.ndarray
    sequence_type: str
    data_set_id: Optional[int] = None
    unique_id: Optional[int] = None
    word_ids: Optional[int] = None


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


@dataclass(frozen=True)
class NamedTargetMaskedLossBase:
    field: Optional[str] = None
    weight: float = 1
    uncertainty_log_sigma_squared_field: Optional[str] = None

    def __post_init__(self):
        if self.uncertainty_log_sigma_squared_field is not None:
            try:
                type(self).uncertainty_weight()
            except RuntimeError:
                raise ValueError('{} does not support uncertainty')

    def apply_weight(self, result):
        is_tuple = isinstance(result, tuple)
        if is_tuple:
            loss = result[0]
        else:
            loss = result
        if isinstance(loss, str):
            assert(loss == 'no_valid_inputs')
        else:
            loss = self.weight * loss
        if is_tuple:
            return (loss,) + result[1:]
        return loss

    def _get_predicted_and_target(self, batch, prediction_dict):
        return prediction_dict[self.field], batch[self.field]

    def __call__(
            self,
            is_eval,
            epoch,
            global_step,
            batch,
            prediction_dict,
            return_detailed=False,
            reduction='mean',
            as_numpy=False,
            apply_weight=True):

        if self.field is None:
            raise ValueError('field is required to be set before loss is called')

        if self.field not in batch:
            if reduction == 'mean' or reduction == 'sum':
                result = 'no_valid_inputs'
            else:
                result = 'no_valid_inputs', 0
            if return_detailed:
                detailed_result = list()
                return result, detailed_result
            else:
                return result

        predictions, target = self._get_predicted_and_target(batch, prediction_dict)
        target = target.to(predictions.device)
        mask = self._get_mask(is_eval, epoch, global_step, batch, predictions, target)

        try:
            result, valid_count = self._masked_loss(is_eval, epoch, global_step, mask, predictions, target)
            if self.uncertainty_log_sigma_squared_field is not None:
                if self.uncertainty_log_sigma_squared_field not in prediction_dict:
                    raise ValueError('log_sigma_squared not found in predictions. Looking for field: {}'.format(
                        self.uncertainty_log_sigma_squared_field))
                result = type(self)._uncertainty_loss(prediction_dict[self.uncertainty_log_sigma_squared_field], result)
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

            word_ids = batch[(self.field, 'word_ids')].detach().cpu().numpy() \
                if (self.field, 'word_ids') in batch else None

            batch_mask = mask.detach().cpu().numpy()
            batch_predictions = predictions.detach().cpu().numpy()
            batch_target = target.detach().cpu().numpy()
            batch_mask = np.split(batch_mask, len(batch_mask)) if len(batch_mask) > 0 else batch_mask
            batch_predictions = np.split(batch_predictions, len(batch_predictions)) \
                if len(batch_predictions) > 0 else batch_predictions
            batch_target = np.split(batch_target, len(batch_target)) if len(batch_target) > 0 else batch_target
            word_ids = np.split(word_ids, len(word_ids)) if word_ids is not None and len(word_ids) > 0 else word_ids

            sequence_type = 'sequence' if self._is_sequence_loss() else 'single'

            if example_indices is not None:  # group by the example indices
                sequence_type = 'grouped'
                grouped = dict()
                for i_ex in range(len(example_indices)):
                    m, p, t, ex = batch_mask[i_ex], batch_predictions[i_ex], batch_target[i_ex], example_indices[i_ex]
                    w = word_ids[i_ex] if word_ids is not None else None
                    if ex not in grouped:
                        grouped[ex] = (list(), list(), list(), list())
                    grouped[ex][0].append(np.expand_dims(m, 1))
                    grouped[ex][1].append(np.expand_dims(p, 1))
                    grouped[ex][2].append(np.expand_dims(t, 1))
                    if w is not None:
                        grouped[ex][3].append(np.expand_dims(w, 1))
                batch_mask = list()
                batch_predictions = list()
                batch_target = list()
                word_ids = list() if word_ids is not None else None
                example_indices = [ex for ex in sorted(grouped)]
                for ex in example_indices:
                    batch_mask.append(np.concatenate(grouped[ex][0], axis=1))
                    batch_predictions.append(np.concatenate(grouped[ex][1], axis=1))
                    batch_target.append(np.concatenate(grouped[ex][2], axis=1))
                    if word_ids is not None:
                        word_ids.append(np.concatenate(grouped[ex][3], axis=1))

            detailed_result = list()
            for idx, (example_mask, example_predictions, example_targets) in enumerate(zip(
                    batch_mask, batch_predictions, batch_target)):

                example_word_ids = None
                if word_ids is not None:
                    example_word_ids = word_ids[idx]

                if example_indices is not None:
                    idx = example_indices[idx]

                data_set_id = batch['data_set_id'].detach().cpu().numpy()[idx] \
                    if 'data_set_id' in batch else None
                unique_id = batch['unique_id'].detach().cpu().numpy()[idx] \
                    if 'unique_id' in batch else None
                detailed_result.append(
                    DetailedResult(
                        mask=np.abs(np.squeeze(example_mask, axis=0) - 1) < 1e-4,  # convert to bool
                        prediction=np.squeeze(example_predictions, axis=0),
                        target=np.squeeze(example_targets, axis=0),
                        sequence_type=sequence_type,
                        data_set_id=data_set_id,
                        unique_id=unique_id,
                        word_ids=np.squeeze(example_word_ids, axis=0) if example_word_ids is not None else None))
            return result, detailed_result
        return result

    def _get_mask(self, is_eval, epoch, global_step, batch, predictions, target):
        return ~torch.isnan(target)

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        raise NotImplementedError('{} does not implement _masked_loss'.format(type(self)))

    @classmethod
    def uncertainty_weight(cls) -> float:
        raise RuntimeError('{} does not support uncertainty')

    @classmethod
    def _uncertainty_loss(cls, log_sigma_squared, loss):
        return cls.uncertainty_weight() * (loss / torch.exp(log_sigma_squared)) + log_sigma_squared / 2

    @classmethod
    def _is_sequence_loss(cls):
        return False

    def shape_adjust(self, shape):
        return shape

    @classmethod
    def is_classification_loss(cls):
        raise NotImplementedError('{} does not implement is_classification_loss'.format(cls))


@dataclass(frozen=True)
class _NamedTargetStopWordAwareLoss(NamedTargetMaskedLossBase):
    keep_content: bool = True

    def _get_mask(self, is_eval, epoch, global_step, batch, predictions, target):
        return stop_word_and_target_not_nan_mask(
            self.keep_content,
            target, batch['is_stop'].to(target.device), batch['is_begin_word_pieces'].to(target.device))

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        raise NotImplementedError('{} does not implement _masked_loss'.format(type(self)))

    @classmethod
    def _is_sequence_loss(cls):
        return True

    @classmethod
    def is_classification_loss(cls):
        raise NotImplementedError('{} does not implement is_classification_loss'.format(cls))


@dataclass(frozen=True)
class NamedTargetStopWordAwareMSE(_NamedTargetStopWordAwareLoss):

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        return masked_squared_error(mask, predictions, target)

    @classmethod
    def uncertainty_weight(cls) -> float:
        return 0.5

    @classmethod
    def is_classification_loss(cls):
        return False


@dataclass(frozen=True)
class NamedTargetStopWordAwareMAE(_NamedTargetStopWordAwareLoss):

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        return masked_absolute_error(mask, predictions, target)

    @classmethod
    def is_classification_loss(cls):
        return False


@dataclass(frozen=True)
class NamedTargetStopWordAwareKLeastSE(_NamedTargetStopWordAwareLoss):
    k_function: Callable[[int, int, int], int] = None
    moving_average_decay: float = 0.98

    # use lists so we can modify these
    _accumulator: List[Optional[torch.Tensor]] = dataclasses.field(
        init=False, repr=False, compare=False, default_factory=lambda: [None])
    _active_mask: List[Optional[torch.Tensor]] = dataclasses.field(
        init=False, repr=False, compare=False, default_factory=lambda: [None])

    def __post_init__(self):
        super().__post_init__()
        if self.k_function is None:
            raise ValueError('k_function is required')

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        num_features = int(np.prod(target.size()[2:]))
        k = self.k_function(epoch, global_step, num_features)
        self._accumulator[0], self._active_mask[0], result = k_least_squared_error(
            is_eval, is_sequence=True, k=k, mask=mask, predictions=predictions, target=target,
            accumulator=self._accumulator[0], active_mask=self._active_mask[0],
            moving_average_decay=self.moving_average_decay)
        return result

    @classmethod
    def uncertainty_weight(cls) -> float:
        return 0.5

    @classmethod
    def is_classification_loss(cls):
        return False


@dataclass(frozen=True)
class NamedTargetStopWordAwareKLeastSEEvalUpdate(_NamedTargetStopWordAwareLoss):
    k_function: Callable[[int, int, int], int] = None

    # use lists so we can modify these
    _accumulator: List[Optional[torch.Tensor]] = dataclasses.field(
        init=False, repr=False, compare=False, default_factory=lambda: [None])
    _counts: List[Optional[torch.Tensor]] = dataclasses.field(
        init=False, repr=False, compare=False, default_factory=lambda: [None])
    _top_k_mask: List[Optional[torch.Tensor]] = dataclasses.field(
        init=False, repr=False, compare=False, default_factory=lambda: [None])
    _num_features: List[Optional[int]] = dataclasses.field(
        init=False, repr=False, compare=False, default_factory=lambda: [None])

    def __post_init__(self):
        super().__post_init__()
        if self.k_function is None:
            raise ValueError('k_function is required')

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        if self._num_features[0] is None:
            self._num_features[0] = int(np.prod(target.size()[2:]))
        self._accumulator[0], self._counts[0], result = k_least_squared_error_update_on_eval(
            is_eval, is_sequence=True, mask=mask, predictions=predictions, target=target,
            top_k_mask=self._top_k_mask[0], next_accumulator=self._accumulator[0], next_counts=self._counts[0])
        return result

    def after_eval_batches(self, epoch, global_step):
        k = self.k_function(epoch, global_step, self._num_features[0])
        self._top_k_mask[0] = update_k_least(self._accumulator[0], self._counts[0], k)
        self._accumulator[0] = None
        self._counts[0] = None

    @classmethod
    def uncertainty_weight(cls) -> float:
        return 0.5

    @classmethod
    def is_classification_loss(cls):
        return False


@dataclass(frozen=True)
class NamedTargetStopWordAwareKLeastAE(_NamedTargetStopWordAwareLoss):
    k_function: Callable[[int, int, int], int] = None
    moving_average_decay: float = 0.98

    # use lists so we can modify these
    _accumulator: List[Optional[torch.Tensor]] = dataclasses.field(
        init=False, repr=False, compare=False, default_factory=lambda: [None])
    _active_mask: List[Optional[torch.Tensor]] = dataclasses.field(
        init=False, repr=False, compare=False, default_factory=lambda: [None])

    def __post_init__(self):
        super().__post_init__()
        if self.k_function is None:
            raise ValueError('k_function is required')

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        num_features = int(np.prod(target.size()[2:]))
        k = self.k_function(epoch, global_step, num_features)
        self._accumulator[0], self._active_mask[0], result = k_least_squared_error(
            is_eval, is_sequence=True, k=k, mask=mask, predictions=predictions, target=target,
            accumulator=self._accumulator[0], active_mask=self._active_mask[0],
            moving_average_decay=self.moving_average_decay, use_abs=True)
        return result

    @classmethod
    def is_classification_loss(cls):
        return False


@dataclass(frozen=True)
class NamedTargetStopWordAwareKLeastAEEvalUpdate(_NamedTargetStopWordAwareLoss):
    k_function: Callable[[int, int, int], int] = None

    # use lists so we can modify these
    _accumulator: List[Optional[torch.Tensor]] = dataclasses.field(
        init=False, repr=False, compare=False, default_factory=lambda: [None])
    _counts: List[Optional[torch.Tensor]] = dataclasses.field(
        init=False, repr=False, compare=False, default_factory=lambda: [None])
    _top_k_mask: List[Optional[torch.Tensor]] = dataclasses.field(
        init=False, repr=False, compare=False, default_factory=lambda: [None])
    _num_features: List[Optional[int]] = dataclasses.field(
        init=False, repr=False, compare=False, default_factory=lambda: [None])

    def __post_init__(self):
        super().__post_init__()
        if self.k_function is None:
            raise ValueError('k_function is required')

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        if self._num_features[0] is None:
            self._num_features[0] = int(np.prod(target.size()[2:]))
        self._accumulator[0], self._counts[0], result = k_least_squared_error_update_on_eval(
            is_eval, is_sequence=True, mask=mask, predictions=predictions, target=target,
            top_k_mask=self._top_k_mask[0], next_accumulator=self._accumulator[0],
            next_counts=self._counts[0], use_abs=True)
        return result

    def after_eval_batches(self, epoch, global_step):
        k = self.k_function(epoch, global_step, self._num_features[0])
        self._top_k_mask[0] = update_k_least(self._accumulator[0], self._counts[0], k)
        self._accumulator[0] = None
        self._counts[0] = None

    @classmethod
    def is_classification_loss(cls):
        return False


@dataclass(frozen=True)
class NamedTargetStopWordAwarePearsonDistance(_NamedTargetStopWordAwareLoss):
    should_penalize_scale: bool = False
    axis: int = 1

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        distance, valid_count, var_input, var_target, mean_input, mean_target = masked_pearsons_distance(
            mask, predictions, target, sequence_axis=self.axis)
        loss = distance
        if self.should_penalize_scale:
            loss = loss + (var_input - var_target) ** 2
        return loss, valid_count

    @classmethod
    def is_classification_loss(cls):
        return False


@dataclass(frozen=True)
class NamedTargetStopWordAwareCrossEntropy(_NamedTargetStopWordAwareLoss):
    num_classes: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        if self.num_classes is None:
            raise ValueError('num_classes is required')

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        return masked_cross_entropy(mask, predictions, target)

    def shape_adjust(self, shape):
        return shape + (self.num_classes,)

    @classmethod
    def is_classification_loss(cls):
        return True

    @classmethod
    def uncertainty_weight(cls) -> float:
        return 1.0


@dataclass(frozen=True)
class NamedTargetStopWordAwareNLL(_NamedTargetStopWordAwareLoss):
    num_classes: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        if self.num_classes is None:
            raise ValueError('num_classes is required')

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        return masked_negative_log_likelihood(mask, predictions, target)

    def shape_adjust(self, shape):
        return shape + (self.num_classes,)

    @classmethod
    def is_classification_loss(cls):
        return True

    @classmethod
    def uncertainty_weight(cls) -> float:
        return 1.0


@dataclass(frozen=True)
class NamedTargetStopWordAwareBinaryCrossEntropyWithLogits(_NamedTargetStopWordAwareLoss):
    pos_weight: Optional[float] = None

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        return masked_binary_cross_entropy_with_logits(mask, predictions, target, self.pos_weight)

    @classmethod
    def is_classification_loss(cls):
        return True

    @classmethod
    def uncertainty_weight(cls) -> float:
        return 1.0



@dataclass(frozen=True)
class NamedTargetStopWordAwareBinaryCrossEntropy(_NamedTargetStopWordAwareLoss):
    pos_weight: Optional[float] = None

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        return masked_binary_cross_entropy(mask, predictions, target, self.pos_weight)

    @classmethod
    def is_classification_loss(cls):
        return True

    @classmethod
    def uncertainty_weight(cls) -> float:
        return 1.0


@dataclass(frozen=True)
class NamedTargetStopWordAwareSoftLabelCrossEntropy(_NamedTargetStopWordAwareLoss):

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        return masked_soft_label_cross_entropy(mask, predictions, target)

    @classmethod
    def is_classification_loss(cls):
        return True

    @classmethod
    def uncertainty_weight(cls) -> float:
        return 1.0


@dataclass(frozen=True)
class NamedTargetStopWordAwareSoftLabelNLL(_NamedTargetStopWordAwareLoss):

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        return masked_soft_label_negative_log_likelihood(mask, predictions, target)

    @classmethod
    def is_classification_loss(cls):
        return True

    @classmethod
    def uncertainty_weight(cls) -> float:
        return 1.0


@dataclass(frozen=True)
class NamedTargetSingleMSE(NamedTargetMaskedLossBase):

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        return masked_squared_error(mask, predictions, target)

    @classmethod
    def is_classification_loss(cls):
        return False

    @classmethod
    def uncertainty_weight(cls) -> float:
        return 0.5


@dataclass(frozen=True)
class NamedTargetSingleMAE(NamedTargetMaskedLossBase):

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        return masked_absolute_error(mask, predictions, target)

    @classmethod
    def is_classification_loss(cls):
        return False


@dataclass(frozen=True)
class KLeastSEHalvingEpochs:
    half_life_in_epochs: float
    delay_in_epochs: int = 0
    minimum_k: int = 0
    final_full_epochs_start: Optional[int] = None

    def __call__(self, epoch, global_step, num_features):
        if self.final_full_epochs_start is not None and epoch >= self.final_full_epochs_start:
            return num_features
        epoch = max(0, epoch - self.delay_in_epochs)
        k = int(np.round(np.power(2., -epoch / self.half_life_in_epochs) * num_features))
        return max(k, min(self.minimum_k, num_features))


@dataclass(frozen=True)
class NamedTargetSingleKLeastSE(NamedTargetMaskedLossBase):
    k_function: Callable[[int, int, int], int] = None
    moving_average_decay: float = 0.98

    # use lists so we can modify these
    _accumulator: List[Optional[torch.Tensor]] = dataclasses.field(
        init=False, repr=False, compare=False, default_factory=lambda: [None])
    _active_mask: List[Optional[torch.Tensor]] = dataclasses.field(
        init=False, repr=False, compare=False, default_factory=lambda: [None])

    def __post_init__(self):
        super().__post_init__()
        if self.k_function is None:
            raise ValueError('k_function is required')

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        num_features = int(np.prod(target.size()[1:]))
        k = self.k_function(epoch, global_step, num_features)
        self._accumulator[0], self._active_mask[0], result = k_least_squared_error(
            is_eval, is_sequence=False, k=k, mask=mask, predictions=predictions, target=target,
            accumulator=self._accumulator[0], active_mask=self._active_mask[0],
            moving_average_decay=self.moving_average_decay)
        return result

    @classmethod
    def is_classification_loss(cls):
        return False

    @classmethod
    def uncertainty_weight(cls) -> float:
        return 0.5


@dataclass(frozen=True)
class NamedTargetSingleKLeastSEEvalUpdate(NamedTargetMaskedLossBase):
    k_function: Callable[[int, int, int], int] = None

    # use lists so we can modify these
    _accumulator: List[Optional[torch.Tensor]] = dataclasses.field(
        init=False, repr=False, compare=False, default_factory=lambda: [None])
    _counts: List[Optional[torch.Tensor]] = dataclasses.field(
        init=False, repr=False, compare=False, default_factory=lambda: [None])
    _top_k_mask: List[Optional[torch.Tensor]] = dataclasses.field(
        init=False, repr=False, compare=False, default_factory=lambda: [None])
    _num_features: List[Optional[int]] = dataclasses.field(
        init=False, repr=False, compare=False, default_factory=lambda: [None])

    def __post_init__(self):
        super().__post_init__()
        if self.k_function is None:
            raise ValueError('k_function is required')

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        if self._num_features[0] is None:
            self._num_features[0] = int(np.prod(target.size()[1:]))
        self._accumulator[0], self._counts[0], result = k_least_squared_error_update_on_eval(
            is_eval, is_sequence=False, mask=mask, predictions=predictions, target=target,
            top_k_mask=self._top_k_mask[0], next_accumulator=self._accumulator[0], next_counts=self._counts[0])
        return result

    def after_eval_batches(self, epoch, global_step):
        k = self.k_function(epoch, global_step, self._num_features[0])
        self._top_k_mask[0] = update_k_least(self._accumulator[0], self._counts[0], k)
        self._accumulator[0] = None
        self._counts[0] = None

    @classmethod
    def is_classification_loss(cls):
        return False

    @classmethod
    def uncertainty_weight(cls) -> float:
        return 0.5


@dataclass(frozen=True)
class NamedTargetSingleKLeastAE(NamedTargetMaskedLossBase):
    k_function: Callable[[int, int, int], int] = None
    moving_average_decay: float = 0.98

    # use lists so we can modify these
    _accumulator: List[Optional[torch.Tensor]] = dataclasses.field(
        init=False, repr=False, compare=False, default_factory=lambda: [None])
    _active_mask: List[Optional[torch.Tensor]] = dataclasses.field(
        init=False, repr=False, compare=False, default_factory=lambda: [None])

    def __post_init__(self):
        super().__post_init__()
        if self.k_function is None:
            raise ValueError('k_function is required')

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        num_features = int(np.prod(target.size()[1:]))
        k = self.k_function(epoch, global_step, num_features)
        self._accumulator[0], self._active_mask[0], result = k_least_squared_error(
            is_eval, is_sequence=False, k=k, mask=mask, predictions=predictions, target=target,
            accumulator=self._accumulator[0], active_mask=self._active_mask[0],
            moving_average_decay=self.moving_average_decay, use_abs=True)
        return result

    @classmethod
    def is_classification_loss(cls):
        return False


@dataclass(frozen=True)
class NamedTargetSingleKLeastAEEvalUpdate(NamedTargetMaskedLossBase):
    k_function: Callable[[int, int, int], int] = None

    # use lists so we can modify these
    _accumulator: List[Optional[torch.Tensor]] = dataclasses.field(
        init=False, repr=False, compare=False, default_factory=lambda: [None])
    _counts: List[Optional[torch.Tensor]] = dataclasses.field(
        init=False, repr=False, compare=False, default_factory=lambda: [None])
    _top_k_mask: List[Optional[torch.Tensor]] = dataclasses.field(
        init=False, repr=False, compare=False, default_factory=lambda: [None])
    _num_features: List[Optional[int]] = dataclasses.field(
        init=False, repr=False, compare=False, default_factory=lambda: [None])

    def __post_init__(self):
        super().__post_init__()
        if self.k_function is None:
            raise ValueError('k_function is required')

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        if self._num_features[0] is None:
            self._num_features[0] = int(np.prod(target.size()[1:]))
        self._accumulator[0], self._counts[0], result = k_least_squared_error_update_on_eval(
            is_eval, is_sequence=False, mask=mask, predictions=predictions, target=target,
            top_k_mask=self._top_k_mask[0], next_accumulator=self._accumulator[0], next_counts=self._counts[0],
            use_abs=True)
        return result

    def after_eval_batches(self, epoch, global_step):
        k = self.k_function(epoch, global_step, self._num_features[0])
        self._top_k_mask[0] = update_k_least(self._accumulator[0], self._counts[0], k)
        self._accumulator[0] = None
        self._counts[0] = None

    @classmethod
    def is_classification_loss(cls):
        return False


@dataclass(frozen=True)
class NamedTargetSinglePearsonDistance(NamedTargetMaskedLossBase):
    should_penalize_scale: bool = False
    axis: int = 0

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        distance, valid_count, var_input, var_target, mean_input, mean_target = masked_pearsons_distance(
            mask, predictions, target, sequence_axis=self.axis)
        loss = distance
        if self.should_penalize_scale:
            loss = loss + (var_input - var_target) ** 2
        return loss, valid_count

    @classmethod
    def is_classification_loss(cls):
        return False


@dataclass(frozen=True)
class NamedTargetSingleCrossEntropy(NamedTargetMaskedLossBase):
    num_classes: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        if self.num_classes is None:
            raise ValueError('num_classes is required')

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        return masked_cross_entropy(mask, predictions, target)

    def shape_adjust(self, shape):
        return shape + (self.num_classes,)

    @classmethod
    def uncertainty_weight(cls) -> float:
        return 1.0

    @classmethod
    def is_classification_loss(cls):
        return True


@dataclass(frozen=True)
class NamedTargetSingleNLL(NamedTargetMaskedLossBase):
    num_classes: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        if self.num_classes is None:
            raise ValueError('num_classes is required')

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        return masked_negative_log_likelihood(mask, predictions, target)

    def shape_adjust(self, shape):
        return shape + (self.num_classes,)

    @classmethod
    def uncertainty_weight(cls) -> float:
        return 1.0

    @classmethod
    def is_classification_loss(cls):
        return True


@dataclass(frozen=True)
class NamedTargetSingleSoftLabelCrossEntropy(NamedTargetMaskedLossBase):

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        return masked_soft_label_cross_entropy(mask, predictions, target)

    @classmethod
    def uncertainty_weight(cls) -> float:
        return 1.0

    @classmethod
    def is_classification_loss(cls):
        return True


@dataclass(frozen=True)
class NamedTargetSingleSoftLabelNLL(NamedTargetMaskedLossBase):

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        return masked_soft_label_negative_log_likelihood(mask, predictions, target)

    @classmethod
    def uncertainty_weight(cls) -> float:
        return 1.0

    @classmethod
    def is_classification_loss(cls):
        return True


@dataclass(frozen=True)
class NamedTargetSingleBinaryCrossEntropyWithLogits(NamedTargetMaskedLossBase):
    pos_weight: Optional[float] = None

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        return masked_binary_cross_entropy_with_logits(mask, predictions, target, self.pos_weight)

    @classmethod
    def uncertainty_weight(cls) -> float:
        return 1.0

    @classmethod
    def is_classification_loss(cls):
        return True


@dataclass(frozen=True)
class NamedTargetSingleBinaryCrossEntropy(NamedTargetMaskedLossBase):
    pos_weight: Optional[float] = None

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        return masked_binary_cross_entropy(mask, predictions, target, self.pos_weight)

    @classmethod
    def uncertainty_weight(cls) -> float:
        return 1.0

    @classmethod
    def is_classification_loss(cls):
        return True
