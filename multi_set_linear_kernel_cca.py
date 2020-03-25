import numpy as np
from scipy.linalg import eigh


class MultiSetLinearKernelCCA:

    def __init__(self, num_components=None, regularization=0, positive_definite_eps=1e-4):
        self.num_components = num_components
        self.regularization = regularization
        self.data_list = None
        self.solutions = None
        self.mu = None
        self.sigma = None
        self.positive_definite_eps = positive_definite_eps

    def fit(self, data_iterable):
        # standardize
        self.data_list = list()
        self.mu = list()
        self.sigma = list()
        for d in data_iterable:
            mu = np.mean(d, axis=0, keepdims=True)
            sigma = np.std(d, axis=0, keepdims=True)
            self.mu.append(mu)
            self.sigma.append(sigma)
            self.data_list.append(np.divide(d - mu, sigma, where=sigma >= 1e-12))

        # compute the kernels
        train = list((d @ d.T) for d in self.data_list)
        cross_cov = np.zeros((len(train[0]) * len(train), len(train[0]) * len(train)), dtype=train[0].dtype)
        diagonal = np.zeros_like(cross_cov)
        penalty = np.zeros_like(diagonal)
        for i1 in range(len(train)):
            for i2 in range(i1, len(train)):
                k = train[i1] @ train[i2]
                if i1 == i2:
                    penalty[
                        i1 * len(train[0]):(i1 + 1) * len(train[0]),
                        i1 * len(train[0]):(i1 + 1) * len(train[0])] = train[i1]
                    diagonal[
                        i1 * len(train[0]):(i1 + 1) * len(train[0]),
                        i1 * len(train[0]):(i1 + 1) * len(train[0])] = k
                else:
                    cross_cov[
                        i1 * len(train[0]):(i1 + 1) * len(train[0]),
                        i2 * len(train[0]):(i2 + 1) * len(train[0])] = k
                    cross_cov[
                        i2 * len(train[0]):(i2 + 1) * len(train[0]),
                        i1 * len(train[0]):(i1 + 1) * len(train[0])] = k.T
        # add an eps to diagonal here to make sure we are numerically positive semi-definite
        _, solutions = eigh(
            cross_cov,
            (1 - self.regularization) * diagonal
            + self.regularization * penalty
            + self.positive_definite_eps * np.identity(len(diagonal)))
        self.solutions = np.split(solutions, len(train))
        # note that because we have regularization, the eigenvalues are not necessarily in order
        # of the correlations. Therefore, we compute the correlations and sort
        r = self.correlations()
        # sort r in descending order
        idx_sort = np.argsort(-r)
        self.solutions = list(sol[:, idx_sort] for sol in self.solutions)
        if self.num_components is not None and self.num_components < self.solutions[0].shape[1]:
            self.solutions = list(sol[:, :self.num_components] for sol in self.solutions)

    def transform(self, data_iterable):
        result = list()
        for d, train, sol, mu, sigma in zip(data_iterable, self.data_list, self.solutions, self.mu, self.sigma):
            d = np.divide(d - mu, sigma, where=sigma >= 1e-12)
            d = d @ train.T
            result.append(d @ sol)
        return result

    def correlations(self, data_iterable=None, use_abs=False):
        if data_iterable is None:
            data_iterable = self.data_list
        components = self.transform(data_iterable)
        result = list()
        for i in range(components[0].shape[1]):
            r = np.corrcoef(np.concatenate(list(np.expand_dims(c[:, i], 0) for c in components)), rowvar=True)
            if use_abs:
                r = np.abs(r)
            result.append(np.mean(r[np.triu_indices_from(r, 1)]))
        return np.array(result)

    def fit_transform(self, data_iterable):
        data_iterable = list(data_iterable)
        self.fit(data_iterable)
        return self.transform(data_iterable)


def multi_set_kcca_cv(
        data_set_partitions, regularization_candidates, num_components=None, validation_data=None, show_cv_corr=False):
    num_partitions = None
    for s in data_set_partitions:
        if num_partitions is None:
            num_partitions = len(data_set_partitions[s])
        elif num_partitions != len(data_set_partitions[s]):
            raise ValueError('Data partitions inconsistent across sets')

    if len(regularization_candidates) > 1:
        candidate_correlations = list()
        for regularization_candidate in regularization_candidates:
            fold_correlations = list()
            for hold_out in range(num_partitions):
                train = list()
                validation = list()
                mcca = MultiSetLinearKernelCCA(num_components, regularization_candidate)
                for s in data_set_partitions:
                    train.append(np.concatenate(list(d for i, d in enumerate(data_set_partitions[s]) if i != hold_out)))
                    validation.append(data_set_partitions[s][hold_out])
                mcca.fit(train)
                components = mcca.transform(validation)
                if num_components is None:
                    # use the components where the correlations on the training data are positive
                    train_r = mcca.correlations()
                    components = [c[:, train_r > 0] for c in components]
                mean_correlations = list()
                for i in range(components[0].shape[1]):
                    r = np.corrcoef(np.concatenate(list(np.expand_dims(c[:, i], 0) for c in components)), rowvar=True)
                    mean_correlations.append(np.mean(np.abs(r[np.triu_indices_from(r, 1)])))
                mean_correlations = np.array(mean_correlations)
                if show_cv_corr:
                    print(regularization_candidate, hold_out, mean_correlations)
                fold_correlations.append(np.mean(mean_correlations))
            candidate_correlations.append(np.mean(fold_correlations))
        selected_regularization = np.argmax(candidate_correlations)
    else:
        selected_regularization = 0

    mcca = MultiSetLinearKernelCCA(num_components, regularization_candidates[selected_regularization])
    data_list = [np.concatenate(data_set_partitions[s]) for s in data_set_partitions]
    mcca.fit(data_list)
    components = mcca.transform(data_list)
    components = type(data_set_partitions)((s, components[i]) for i, s in enumerate(data_set_partitions))
    validation_components = None
    if validation_data is not None:
        validation_components = mcca.transform([validation_data[s] for s in data_set_partitions])
        validation_components = type(validation_data)(
            (s, validation_components[i]) for i, s in enumerate(data_set_partitions))
    if num_components is None:
        r = mcca.correlations()
        for s in components:
            components[s] = components[s][:, r > 0]
            if validation_components is not None:
                validation_components[s] = validation_components[s][:, r > 0]
    if validation_components is not None:
        return components, validation_components
    return components
