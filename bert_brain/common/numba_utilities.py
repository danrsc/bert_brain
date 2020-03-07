import numpy as np
from numba import njit, prange


__all__ = ['nandetrend']


@njit(parallel=True)
def _nandetrend(y, x):
    result = np.copy(y)
    for i in prange(y.shape[1]):
        target = y[:, i]
        is_valid = np.logical_not(np.logical_or(np.isnan(x[:, 0]), np.isnan(target)))
        source = x[is_valid]
        target_ = target[is_valid]
        solution, error, rank, singular_values = np.linalg.lstsq(source, target_)
        residuals = target_ - np.dot(source, solution)
        # set the result to the residual
        k = 0
        for j in range(len(is_valid)):
            if is_valid[j]:
                result[j, i] = residuals[k]
                k += 1
    return result


def nandetrend(y, x=None):
    shape = y.shape
    y = np.reshape(y, (y.shape[0], -1))
    if x is None:
        x = np.arange(len(y)).astype(y.dtype)
    # concatenate seems to sometimes crash in numba, so we add the bias term here
    x = np.concatenate((np.expand_dims(x, 1), np.expand_dims(np.ones_like(x), 1)), axis=1)
    return np.reshape(_nandetrend(y, x), shape)
