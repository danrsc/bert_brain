from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.kernel_ridge import KernelRidge


def _kernel_ridge_score(x_train, y_train, x_validation, y_validation, alpha):
    model = KernelRidge(alpha, kernel='precomputed')
    model.fit(x_train, y_train)
    y_hat = model.predict(x_validation)
    mse = np.mean((y_hat - y_validation) ** 2, axis=0)
    return mse


def kernel_ridge_cv(
        x_train_partitions, y_train_partitions, x_validation, y_validation, alpha_candidates, max_workers=None):

    scores = dict()
    for hold_out in range(len(x_train_partitions)):
        x_train = np.concatenate(list(r for i, r in enumerate(x_train_partitions) if i != hold_out))
        x_validation_internal = x_train_partitions[hold_out]
        y_train = np.concatenate(list(r for i, r in enumerate(y_train_partitions) if i != hold_out))
        y_validation_internal = y_train_partitions[hold_out]

        train_mu = np.mean(x_train, axis=0, keepdims=True)
        train_sigma = np.std(x_train, axis=0, keepdims=True)
        indicator_valid = train_sigma[0] > 1e-12
        x_train = x_train[:, indicator_valid]
        x_validation_internal = x_validation_internal[:, indicator_valid]
        train_mu = train_mu[:, indicator_valid]
        train_sigma = train_sigma[:, indicator_valid]

        x_train = (x_train - train_mu) / train_sigma
        x_validation_internal = (x_validation_internal - train_mu) / train_sigma

        y_mu = np.mean(y_train, axis=0, keepdims=True)
        y_sigma = np.std(y_train, axis=0, keepdims=True)

        y_train = np.divide(y_train - y_mu, y_sigma, where=y_sigma > 1e-12)
        y_validation_internal = np.divide(y_validation_internal - y_mu, y_sigma, where=y_sigma > 1e-12)

        # form the kernels
        x_validation_internal = x_validation_internal @ x_train.T
        x_train = x_train @ x_train.T

        if max_workers is not None and max_workers < 2:
            for index_alpha, score in enumerate(map(
                    _kernel_ridge_score,
                    [x_train for _ in alpha_candidates],
                    [y_train for _ in alpha_candidates],
                    [x_validation_internal for _ in alpha_candidates],
                    [y_validation_internal for _ in alpha_candidates],
                    alpha_candidates)):
                if index_alpha not in scores:
                    scores[index_alpha] = list()
                scores[index_alpha].append(np.expand_dims(score, 0))
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                for index_alpha, score in enumerate(ex.map(
                        _kernel_ridge_score,
                        [x_train for _ in alpha_candidates],
                        [y_train for _ in alpha_candidates],
                        [x_validation_internal for _ in alpha_candidates],
                        [y_validation_internal for _ in alpha_candidates],
                        alpha_candidates)):
                    if index_alpha not in scores:
                        scores[index_alpha] = list()
                    scores[index_alpha].append(np.expand_dims(score, 0))

    min_loss = None
    selected_alpha = None
    for index_alpha in scores:
        loss = np.mean(np.concatenate(scores[index_alpha]), axis=0)
        if min_loss is None:
            selected_alpha = alpha_candidates[index_alpha]
            min_loss = loss
        else:
            selected_alpha = np.where(loss < min_loss, alpha_candidates[index_alpha], selected_alpha)
            min_loss = np.minimum(loss, min_loss)

    x_train = np.concatenate(x_train_partitions)
    y_train = np.concatenate(y_train_partitions)

    train_mu = np.mean(x_train, axis=0, keepdims=True)
    train_sigma = np.std(x_train, axis=0, keepdims=True)
    indicator_valid = train_sigma[0] > 1e-12
    x_train = x_train[:, indicator_valid]
    if x_validation is None:
        x_validation = x_train
    else:
        x_validation = x_validation[:, indicator_valid]
    train_mu = train_mu[:, indicator_valid]
    train_sigma = train_sigma[:, indicator_valid]

    x_train = (x_train - train_mu) / train_sigma
    x_validation = (x_validation - train_mu) / train_sigma

    y_mu = np.mean(y_train, axis=0, keepdims=True)
    y_sigma = np.std(y_train, axis=0, keepdims=True)

    y_train = np.divide(y_train - y_mu, y_sigma, where=y_sigma > 1e-12)
    if y_validation is None:
        y_validation = y_train
    else:
        y_validation = np.divide(y_validation - y_mu, y_sigma, where=y_sigma > 1e-12)

    x_validation = x_validation @ x_train.T
    x_train = x_train @ x_train.T

    model = KernelRidge(selected_alpha, kernel='precomputed')
    model.fit(x_train, y_train)
    y_hat = model.predict(x_validation)
    mse = np.mean((y_hat - y_validation) ** 2, axis=0)

    return mse, y_hat, selected_alpha
