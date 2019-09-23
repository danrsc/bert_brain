import warnings

import numpy as np
from tqdm import trange


def one_sample_permutation_test(
        sample_predictions,
        sample_target,
        value_fn,
        num_contiguous_examples=10,
        num_permutation_samples=1000,
        unique_ids=None,
        side='both'):

    if side not in ['both', 'less', 'greater']:
        raise ValueError('side must be one of: \'both\', \'less\', \'greater\'')

    def _sort(predictions, targets, ids):
        if ids is None:
            return predictions, targets, ids
        ids = np.asarray(ids)
        sort_order = np.argsort(ids)
        return predictions[sort_order], targets[sort_order], ids[sort_order]

    sample_predictions, sample_target, unique_ids = _sort(sample_predictions, sample_target, unique_ids)

    keep_length = int(np.ceil(len(sample_predictions) / num_contiguous_examples)) * num_contiguous_examples
    sample_predictions = sample_predictions[:keep_length]
    sample_target = sample_target[:keep_length]
    true_values = value_fn(sample_predictions, sample_target)
    abs_true_values = np.abs(true_values) if side == 'both' else None

    sample_target = np.reshape(
        sample_target,
        (sample_target.shape[0] // num_contiguous_examples, num_contiguous_examples) + sample_target.shape[1:])

    count_as_extreme = np.zeros(abs_true_values.shape, np.int64)
    for _ in trange(num_permutation_samples, desc='Permutation'):
        indices_target = np.random.permutation(len(sample_target))
        permuted_target = np.reshape(
            sample_target[indices_target], (sample_predictions.shape[0],) + sample_target.shape[2:])
        permuted_values = value_fn(sample_predictions, permuted_target)
        if side == 'less':
            as_extreme = np.where(np.less_equal(permuted_values, true_values), 1, 0)
        elif side == 'greater':
            as_extreme = np.where(np.greater_equal(permuted_values, true_values), 1, 0)
        else:
            assert(side == 'both')
            as_extreme = np.where(np.greater_equal(permuted_values, abs_true_values), 1, 0)
        count_as_extreme += as_extreme

    p_values = count_as_extreme / num_permutation_samples
    return p_values, true_values


def two_sample_permutation_test(
        sample_a_values,
        sample_b_values,
        num_contiguous_examples=10,
        num_permutation_samples=1000,
        unique_ids_a=None,
        unique_ids_b=None):

    def _sort(values, ids):
        if ids is None:
            return values, ids
        ids = np.asarray(ids)
        sort_order = np.argsort(ids)
        return values[sort_order], ids[sort_order]

    sample_a_values, unique_ids_a = _sort(sample_a_values, unique_ids_a)
    sample_b_values, unique_ids_b = _sort(sample_b_values, unique_ids_b)

    if unique_ids_a is not None and unique_ids_b is not None:
        if not np.array_equal(unique_ids_a, unique_ids_b):
            raise ValueError('Ids do not match between unique_ids_a and unique_ids_b')

    def _contiguous_values(s):
        fill_length = int(np.ceil(len(s) / num_contiguous_examples)) * num_contiguous_examples
        temp = np.full((fill_length,) + s.shape[1:], np.nan)
        temp[:len(s)] = s
        temp = np.reshape(temp, (len(temp) // num_contiguous_examples, num_contiguous_examples) + temp.shape[1:])
        return np.nanmean(temp, axis=1)

    sample_a_values = _contiguous_values(sample_a_values)
    sample_b_values = _contiguous_values(sample_b_values)

    true_difference = np.mean(sample_a_values - sample_b_values, axis=0)
    abs_true_difference = np.abs(true_difference)

    all_values = np.concatenate([sample_a_values, sample_b_values], axis=0)

    count_greater_equal = np.zeros(true_difference.shape, np.int64)
    for _ in trange(num_permutation_samples, desc='Permutation'):
        permuted = np.random.permutation(all_values)
        first, second = np.split(permuted, 2)
        permutation_difference = np.abs(np.mean(first - second, axis=0))
        greater_equal = np.where(np.greater_equal(permutation_difference, abs_true_difference), 1, 0)
        count_greater_equal += greater_equal
    p_values = count_greater_equal / num_permutation_samples
    return p_values, true_difference


def wilcoxon_axis(x, y=None, zero_method="wilcox", correction=False):
    # copied from scipy.stats with adjustments so we can apply it along axis=0
    from scipy.stats import distributions
    from scipy.stats.mstats import rankdata
    """
    Calculate the Wilcoxon signed-rank test.

    The Wilcoxon signed-rank test tests the null hypothesis that two
    related paired samples come from the same distribution. In particular,
    it tests whether the distribution of the differences x - y is symmetric
    about zero. It is a non-parametric version of the paired T-test.

    Parameters
    ----------
    x : array_like
        The first set of measurements.
    y : array_like, optional
        The second set of measurements.  If `y` is not given, then the `x`
        array is considered to be the differences between the two sets of
        measurements.
    zero_method : string, {"pratt", "wilcox", "zsplit"}, optional
        "pratt":
            Pratt treatment: includes zero-differences in the ranking process
            (more conservative)
        "wilcox":
            Wilcox treatment: discards all zero-differences
        "zsplit":
            Zero rank split: just like Pratt, but spliting the zero rank
            between positive and negative ones
    correction : bool, optional
        If True, apply continuity correction by adjusting the Wilcoxon rank
        statistic by 0.5 towards the mean value when computing the
        z-statistic.  Default is False.

    Returns
    -------
    T : float
        The sum of the ranks of the differences above or below zero, whichever
        is smaller.
    p-value : float
        The two-sided p-value for the test.

    Notes
    -----
    Because the normal approximation is used for the calculations, the
    samples used should be large.  A typical rule is to require that
    n > 20.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test

    """

    if zero_method not in ["wilcox", "pratt", "zsplit"]:
        raise ValueError("Zero method should be either 'wilcox' \
                          or 'pratt' or 'zsplit'")

    if y is None:
        d = x
    else:
        x, y = map(np.asarray, (x, y))
        if len(x) != len(y):
            raise ValueError('Unequal N in wilcoxon.  Aborting.')
        d = x-y

    d_shape = d.shape
    d = np.reshape(d, (d.shape[0], -1))

    if zero_method == "wilcox":
        d = np.where(np.not_equal(d, 0), d, np.nan)  # Keep all non-zero differences

    count = np.sum(np.logical_not(np.isnan(d)), axis=0)
    if np.any(count < 10):
        warnings.warn("Warning: sample size too small for normal approximation.")
    ranked = rankdata(np.ma.masked_invalid(np.abs(d[:, count > 0])), axis=0)
    r = np.full(d.shape, np.nan, ranked.dtype)
    r[:, count > 0] = ranked
    r = np.where(r == 0, np.nan, r)
    r_plus = np.nansum((d > 0) * r, axis=0)
    r_minus = np.nansum((d < 0) * r, axis=0)
    if zero_method == "zsplit":
        r_zero = np.nansum((d == 0) * r, axis=0)
        r_plus += r_zero / 2.
        r_minus += r_zero / 2.

    T = np.minimum(r_plus, r_minus)
    mn = count*(count + 1.) * 0.25
    se = count*(count + 1.) * (2. * count + 1.)

    if zero_method == "pratt":
        r = np.where(d == 0, np.nan, r)

    flat_r = np.reshape(r, (r.shape[0], -1, 1))
    column_id = np.tile(np.reshape(np.arange(flat_r.shape[1]), (1, -1, 1)), (r.shape[0], 1, 1))
    flat_r_with_column = np.reshape(np.concatenate([column_id, flat_r], axis=2), (-1, 2))

    repeats_with_column, repnum = np.unique(flat_r_with_column, return_counts=True, axis=0)
    repeats_with_column = repeats_with_column[repnum > 1]
    repnum = repnum[repnum > 1]
    if len(repnum) != 0:
        column_id = np.asarray(np.round(repeats_with_column[:, 0]), dtype=np.int64)
        weights = repnum * (repnum * repnum - 1)
        weights = np.asarray(weights, dtype=np.float64)
        repeat_correction = 0.5 * np.bincount(column_id, weights=weights)
        column_repeat_correction = np.zeros(flat_r.shape[1], se.dtype)
        column_repeat_correction[:len(repeat_correction)] += repeat_correction
        column_repeat_correction = np.reshape(column_repeat_correction, se.shape)
        # Correction for repeated elements.
        se -= column_repeat_correction

    se = np.sqrt(se / 24)
    correction = 0.5 * int(bool(correction)) * np.sign(T - mn)
    z = (T - mn - correction) / se
    prob = 2. * distributions.norm.sf(np.abs(z))
    return np.reshape(T, d_shape[1:]), np.reshape(prob, d_shape[1:])


def fdr_correction(p_values, alpha=0.05, method='by', axis=None):
    """
    This covers Benjamini/Hochberg for independent or positively correlated and
    Benjamini/Yekutieli for general or negatively correlated tests.
    Modified from the code at https://www.statsmodels.org/dev/_modules/statsmodels/stats/multitest.html

    Args:
        p_values: The p_values to correct.
        alpha: The error rate to correct the p-values with.
        method: one of by (for Benjamini/Yekutieli) or bh for Benjamini/Hochberg
        axis: Which axis of p_values to apply the correction along. If None, p_values is flattened.

    Returns:
        indicator_alternative: An boolean array with the same shape as p_values_corrected that is True where
            the null hypothesis should be rejected
        p_values_corrected: The p_values corrected for FDR. Same shape as p_values
    """
    p_values = np.asarray(p_values)

    shape = p_values.shape
    if axis is None:
        p_values = np.reshape(p_values, -1)
        axis = 0

    indices_sorted = np.argsort(p_values, axis=axis)
    p_values = np.take_along_axis(p_values, indices_sorted, axis=axis)

    correction_factor = np.arange(1, p_values.shape[axis] + 1) / p_values.shape[axis]
    if method == 'bh':
        pass
    elif method == 'by':
        c_m = np.sum(1 / np.arange(1, p_values.shape[axis] + 1), axis=axis, keepdims=True)
        correction_factor = correction_factor / c_m
    else:
        raise ValueError('Unrecognized method: {}'.format(method))

    # set everything left of the maximum qualifying p-value
    indicator_alternative = p_values <= correction_factor * alpha
    indices_all = np.reshape(
        np.arange(indicator_alternative.shape[axis]),
        (1,) * axis + (indicator_alternative.shape[axis],) + (1,) * (len(indicator_alternative.shape) - 1 - axis))
    indices_max = np.max(np.where(indicator_alternative, indices_all, np.nan), axis=axis, keepdims=True)
    indicator_alternative = indices_all <= indices_max
    del indices_all

    p_values = np.clip(
        np.take(
            np.minimum.accumulate(
                np.take(p_values / correction_factor, np.arange(p_values.shape[axis] - 1, -1, -1), axis=axis),
                axis=axis),
            np.arange(p_values.shape[axis] - 1, -1, -1),
            axis=axis),
        low=0,
        high=1)

    indices_sorted = np.argsort(indices_sorted, axis=axis)
    p_values = np.take_along_axis(p_values, indices_sorted, axis=axis)
    indicator_alternative = np.take_along_axis(indicator_alternative, indices_sorted, axis=axis)

    return np.reshape(indicator_alternative, shape), np.reshape(p_values, shape)


def sample_differences(
        sample_a_values,
        sample_b_values,
        num_contiguous_examples=10,
        unique_ids_a=None,
        unique_ids_b=None):

    def _sort(values, ids):
        if ids is None:
            return values, ids
        ids = np.asarray(ids)
        sort_order = np.argsort(ids)
        return values[sort_order], ids[sort_order]

    sample_a_values, unique_ids_a = _sort(sample_a_values, unique_ids_a)
    sample_b_values, unique_ids_b = _sort(sample_b_values, unique_ids_b)

    if unique_ids_a is not None and unique_ids_b is not None:
        if not np.array_equal(unique_ids_a, unique_ids_b):
            raise ValueError('Ids do not match between unique_ids_a and unique_ids_b')

    def _contiguous_values(s):
        fill_length = int(np.ceil(len(s) / num_contiguous_examples)) * num_contiguous_examples
        temp = np.full((fill_length,) + s.shape[1:], np.nan)
        temp[:len(s)] = s
        temp = np.reshape(temp, (len(temp) // num_contiguous_examples, num_contiguous_examples) + temp.shape[1:])
        return np.nanmean(temp, axis=1)

    sample_a_values = _contiguous_values(sample_a_values)
    sample_b_values = _contiguous_values(sample_b_values)

    return sample_a_values - sample_b_values
