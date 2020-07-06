import numpy as np
from numba import njit, prange


__all__ = ['nandetrend', 'auto_regression_residual', 'modified_gram_schmidt', 'batch_psim', 'batch_csim']


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


@njit(parallel=True)
def _batch_psim_cosine(x):
    result = np.empty((x.shape[0], x.shape[1], x.shape[1]), dtype=np.float64)
    for i in prange(x.shape[0]):
        z = x[i]
        norms = np.sqrt(np.sum(np.square(z), axis=-1))
        for j in range(len(z)):
            for k in range(j, len(z)):
                dot = np.vdot(z[j], z[k])
                cosine = dot / (norms[j] * norms[k])
                result[i, j, k] = cosine
                result[i, k, j] = cosine

    return result


@njit(parallel=True)
def _batch_csim_cosine(x, y):
    result = np.empty((x.shape[0], x.shape[1], y.shape[1]), dtype=np.float64)
    for i in prange(x.shape[0]):
        z = x[i]
        w = y[i]
        norms_z = np.sqrt(np.sum(np.square(z), axis=-1))
        norms_w = np.sqrt(np.sum(np.square(w), axis=-1))
        for j in range(len(z)):
            for k in range(len(w)):
                dot = np.vdot(z[j], w[k])
                result[i, j, k] = dot / (norms_z[j] * norms_w[k])

    return result


@njit
def _mu_sigma_last_keepdim(x, ddof=1):
    mu = np.zeros(x.shape[:-1], dtype=x.dtype)
    sum_sq_diff = np.zeros(x.shape[:-1], dtype=x.dtype)
    count = 0
    for i in range(x.shape[-1]):
        count += 1
        v = x[..., i]
        delta = v - mu
        mu += delta / count
        delta_2 = v - mu
        sum_sq_diff += delta * delta_2

    mu = np.expand_dims(mu, -1)
    sum_sq_diff = np.expand_dims(sum_sq_diff, -1)

    if count - ddof < 1:
        return mu, np.full_like(mu, np.nan)
    return mu, np.sqrt(sum_sq_diff / (count - ddof))


@njit
def _center_last(x):
    return x - np.expand_dims(np.sum(x, axis=-1) / x.shape[-1], -1)


@njit
def _standardize_last(x, ddof=1):
    mu, sigma = _mu_sigma_last_keepdim(x, ddof=ddof)
    if x.shape[-1] - ddof < 1:
        return x - mu
    return (x - mu) / sigma


@njit(parallel=True)
def _batch_psim_corr(x, ddof=1):
    result = np.empty((x.shape[0], x.shape[1], x.shape[1]), dtype=np.float64)
    for i in prange(x.shape[0]):
        z = _standardize_last(x[i], ddof=ddof)
        for j in range(len(z)):
            for k in range(j, len(z)):
                r = np.vdot(z[j], z[k]) / (x.shape[-1] - ddof)
                result[i, j, k] = r
                result[i, k, j] = r

    return result


@njit(parallel=True)
def _batch_psim_oovariance_scaled_corr(x, cov):
    result = np.empty((x.shape[0], x.shape[1], x.shape[1]), dtype=np.float64)
    for i in prange(x.shape[0]):
        z = _center_last(x[i])
        numerator = np.dot(z, np.transpose(np.dot(z, cov[i])))
        for j in range(len(z)):
            for k in range(j, len(z)):
                r = numerator[j, k] / np.sqrt(numerator[j, j] * numerator[k, k])
                result[i, j, k] = r
                result[i, k, j] = r

    return result


@njit(parallel=True)
def _batch_csim_corr(x, y, ddof=1):
    result = np.empty((x.shape[0], x.shape[1], y.shape[1]), dtype=np.float64)
    for i in prange(x.shape[0]):
        z = _standardize_last(x[i], ddof=ddof)
        w = _standardize_last(y[i], ddof=ddof)
        for j in range(len(z)):
            for k in range(len(w)):
                result[i, j, k] = np.vdot(z[j], w[k]) / (x.shape[-1] - ddof)

    return result


@njit(parallel=True)
def _batch_csim_oovariance_scaled_corr(x, y, cov):
    result = np.empty((x.shape[0], x.shape[1], y.shape[1]), dtype=np.float64)
    for i in prange(x.shape[0]):
        z = _center_last(x[i])
        w = _center_last(y[i])
        scaled_w = np.dot(w, cov[i])
        denom_z = np.sqrt(np.sum(z * (np.dot(z, cov[i])), axis=-1))
        denom_w = np.sqrt(np.sum(w * scaled_w, axis=-1))
        numerator = z @ np.transpose(scaled_w)
        for j in range(len(z)):
            for k in range(len(w)):
                result[i, j, k] = numerator[j, k] / (denom_z[j] * denom_w[k])

    return result


def batch_psim(x, metric='cosine', ddof=1, cov=None):
    x = np.asarray(x)
    if np.ndim(x) != 3:
        raise ValueError('Expected shape (batch, variables, components)')
    if metric == 'cosine':
        return _batch_psim_cosine(x)
    elif metric == 'correlation':
        return _batch_psim_corr(x, ddof=ddof)
    elif metric == 'covariance_scaled_correlation':
        if cov is None:
            raise ValueError('Covariance must be passed in for covariance_scaled_correlation')
        return _batch_psim_oovariance_scaled_corr(x, cov)
    else:
        raise ValueError('Unsupported metric: {}'.format(metric))


def batch_csim(x, y, metric='cosine', ddof=1, cov=None):
    x = np.asarray(x)
    y = np.asarray(y)
    if np.ndim(x) != 3 or np.ndim(y) != 3:
        raise ValueError('Expected shape (batch, variables, components)')
    if x.shape[0] != y.shape[0]:
        raise ValueError('Mismatched shape along dimension 0')
    if x.shape[-1] != y.shape[-1]:
        raise ValueError('Mismatched shape along dimension -1')
    if metric == 'cosine':
        return _batch_csim_cosine(x, y)
    elif metric == 'correlation':
        return _batch_csim_corr(x, y, ddof=ddof)
    elif metric == 'covariance_scaled_correlation':
        if cov is None:
            raise ValueError('Covariance must be passed in for covariance_scaled_correlation')
        return _batch_csim_oovariance_scaled_corr(x, y, cov)
    else:
        raise ValueError('Unsupported metric: {}'.format(metric))
