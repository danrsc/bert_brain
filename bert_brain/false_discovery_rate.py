import numpy as np


__all__ = ['fdr_correction', 'negative_tail_fdr_threshold']


def negative_tail_fdr_threshold(x, chance_level, alpha=0.05, axis=-1):
    """
    The idea of this is to assume that the noise distribution around the known chance level is symmetric. We can then
    estimate how many of the values at a given level above the chance level are due to noise based on how many values
    there are at the symmetric below chance level.
    Args:
        x: The data
        chance_level: The known chance level for this metric.
            For example, if the metric is correlation, this could be 0.
        alpha: Significance level
        axis: Which axis contains the distribution of values

    Returns:
        The threshold at which only alpha of the values are due to noise, according to this estimation method
    """
    noise_values = np.where(x <= chance_level, x, np.inf)
    # sort ascending, i.e. from most extreme to least extreme
    noise_values = np.sort(noise_values, axis=axis)
    noise_values = np.where(np.isfinite(noise_values), noise_values, np.nan)

    mixed_values = np.where(x > chance_level, x, -np.inf)
    # sort descending, i.e. from most extreme to least extreme
    mixed_values = np.sort(-mixed_values, axis=axis)
    mixed_values = np.where(np.isfinite(mixed_values), mixed_values, np.nan)

    # arange gives the number of values which are more extreme in a sorted array
    num_more_extreme = np.arange(x.shape[axis])
    # if we take these to be the mixed counts, then multiplying by alpha (after including the value itself)
    # gives us the maximum noise counts, which we can use as an index
    # we also add 1 at the end to include the item at that level
    noise_counts = np.ceil(alpha * (num_more_extreme + 1)).astype(np.intp) + 1

    # filter out illegal indexes
    indicator_valid = noise_counts < noise_values.shape[axis]

    noise_values_at_counts = np.take(noise_values, noise_counts[indicator_valid], axis=axis)
    mixed_values_at_counts = np.take(mixed_values, np.arange(mixed_values.shape[axis])[indicator_valid], axis=axis)

    # if the (abs) mixed value is greater than the (abs) noise value, we would have to move to the left on the noise
    # counts to get to the mixed value (i.e. the threshold), which is in the direction of decreasing counts. Therefore
    # at this threshold, the fdr is less than alpha
    noise_values_at_counts = np.abs(noise_values_at_counts - chance_level)
    mixed_values_at_counts = np.abs(mixed_values_at_counts - chance_level)
    thresholds = np.where(mixed_values_at_counts >= noise_values_at_counts, mixed_values_at_counts, np.nan)
    # take the minimum value where this holds
    thresholds = np.nanmin(thresholds, axis=axis)
    return thresholds


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
    indices_max = np.nanmax(np.where(indicator_alternative, indices_all, np.nan), axis=axis, keepdims=True).astype(int)
    indicator_alternative = indices_all <= indices_max
    del indices_all

    p_values = np.clip(
        np.take(
            np.minimum.accumulate(
                np.take(p_values / correction_factor, np.arange(p_values.shape[axis] - 1, -1, -1), axis=axis),
                axis=axis),
            np.arange(p_values.shape[axis] - 1, -1, -1),
            axis=axis),
        a_min=0,
        a_max=1)

    indices_sorted = np.argsort(indices_sorted, axis=axis)
    p_values = np.take_along_axis(p_values, indices_sorted, axis=axis)
    indicator_alternative = np.take_along_axis(indicator_alternative, indices_sorted, axis=axis)

    return np.reshape(indicator_alternative, shape), np.reshape(p_values, shape)
