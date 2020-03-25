import numpy as np
from numba import njit, prange


__all__ = ['nandetrend', 'auto_regression_residual', 'modified_gram_schmidt']


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


@njit
def modified_gram_schmidt(x, min_norm=1e-12):
    kappa = 0.5
    result = np.zeros(x.shape, dtype=x.dtype)
    for k in range(x.shape[1]):
        q = x[:, k]
        orig_norm = np.linalg.norm(q, ord=2)
        for j in range(k):
            q = q - np.vdot(result[:, j], q) * result[:, j]
        norm = np.linalg.norm(q, ord=2)
        if norm >= min_norm:
            if norm / orig_norm <= kappa:
                # re-orthogonalize
                for j in range(k):
                    q = q - np.vdot(result[:, j], q) * result[:, j]
                norm = np.linalg.norm(q, ord=2)
            if norm >= min_norm:
                result[:, k] = q / norm
    return result


@njit(parallel=True)
def _auto_regression_residual(x, max_lag):
    result = np.empty(x.shape, dtype=np.float64)
    for i in prange(x.shape[1]):
        x_ = x[:, i]
        source = np.zeros((len(x_), max_lag), dtype=x_.dtype)
        for j in range(1, max_lag + 1):
            source[j:, j - 1] = x_[:-j]
        solution, error, rank, singular_values = np.linalg.lstsq(source, x_)
        result[:, i] = x_ - np.dot(source, solution)
    return result


def auto_regression_residual(x, max_lag):
    return np.reshape(_auto_regression_residual(np.reshape(x, (len(x), -1)), max_lag), x.shape)