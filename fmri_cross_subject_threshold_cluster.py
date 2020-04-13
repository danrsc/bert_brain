from collections import OrderedDict
import warnings

import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import FastICA, PCA
from tqdm.auto import tqdm
from bottleneck import nanrankdata

from kernel_ridge_cv_module import kernel_ridge_cv
from multi_set_linear_kernel_cca import multi_set_kcca_cv
from bert_brain import modified_gram_schmidt


def eps_standardize(x, axis=0, eps=1e-12):
    std = np.nanstd(x, axis=axis, keepdims=True)
    return np.where(std < eps, np.nan, np.divide(x - np.nanmean(x, axis=axis, keepdims=True), std, where=std >= eps))


def _tuple_key(k):
    if isinstance(k, str):
        return k,
    return k


def _progress_iterate(progress, iterable):
    for item in iterable:
        yield item
        progress.update()


class _ValidArrHelper:

    def __init__(self, data):
        num_partitions = None
        for s in data:
            if num_partitions is None:
                num_partitions = len(data[s])
            elif num_partitions != len(data[s]):
                raise ValueError('Data partitions inconsistent across subjects')

        valid_rows = dict()
        valid_columns = dict()
        for index_partition in range(num_partitions):
            for subject in data:
                is_finite = np.isfinite(data[subject][index_partition])
                current_valid_rows = np.any(is_finite, axis=1)
                current_valid_columns = np.any(is_finite, axis=0)
                if np.any(np.logical_not(is_finite[current_valid_rows][:, current_valid_columns])):
                    raise ValueError('Could not find finite sub-matrix')
                if index_partition not in valid_rows:
                    valid_rows[index_partition] = current_valid_rows
                elif not np.array_equal(valid_rows[index_partition], current_valid_rows):
                    raise ValueError('All subjects must have the same nan rows')
                if subject not in valid_columns:
                    valid_columns[subject] = current_valid_columns
                else:
                    valid_columns[subject] = current_valid_columns
        self.valid_rows = valid_rows
        self.valid_columns = valid_columns

    @property
    def num_partitions(self):
        return len(self.valid_rows)

    def row_count(self, index_partition):
        return len(self.valid_rows[index_partition])

    def compress(self, arr, index_partition, subject):
        return arr[self.valid_rows[index_partition]][:, self.valid_columns[subject]]

    def fill(self, arr, index_partition, subject):
        if np.ndim(arr) == 1:
            x = np.full(len(self.valid_columns[subject]), np.nan)
            x[self.valid_columns[subject]] = arr
            arr = x
        else:
            x = np.full((len(arr), len(self.valid_columns[subject])), np.nan)
            x[:, self.valid_columns[subject]] = arr
            arr = x
            x = np.full((len(self.valid_rows[index_partition]), arr.shape[1]), np.nan)
            x[self.valid_rows[index_partition]] = arr
            arr = x
        return arr


def residual_from_twice_masked_data_with_ica_input(
        data,
        mask_x,
        mask_y,
        ica_components=10,
        max_iter=5000,
        tol=5e-4):
    d = np.reshape(data, (len(data), -1))
    mask_x = np.reshape(mask_x, -1)
    mask_y = np.reshape(mask_y, -1)
    bad_rows = np.any(np.logical_not(np.isfinite(d)), axis=1)
    if np.any(np.logical_and(bad_rows, np.any(np.isfinite(d), axis=1))):
        raise ValueError('Rows must have all finite values or all infinite values')
    valid_rows = np.logical_not(bad_rows)
    d = d[valid_rows]
    x = d[:, mask_x]
    y = d[:, mask_y]
    x = eps_standardize(x)
    ica = FastICA(n_components=ica_components, max_iter=max_iter, tol=tol)
    x = ica.fit_transform(x)
    x = np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1)
    solution, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    y_hat = np.dot(x, solution)

    y_hat_ = np.full((len(valid_rows), y_hat.shape[1]), np.nan, dtype=y_hat.dtype)
    y_hat_[valid_rows] = y_hat
    y_hat = y_hat_
    d = np.copy(data)
    d = np.reshape(d, (len(d), -1))
    d[:, mask_y] = d[:, mask_y] - y_hat
    return np.reshape(d, data.shape)


def multi_set_ica(
        fmri_subject_data_partitions, indicator_train_dict, hold_out, num_components,
        max_iter=200, tol=1e-4):

    valid_rows = None
    for s in fmri_subject_data_partitions:
        v = [_valid_rows(r) for r in fmri_subject_data_partitions[s]]
        if valid_rows is None:
            valid_rows = v
        else:
            if len(valid_rows) != len(v):
                raise ValueError('Inconsistent valid rows across subjects')
            for have, r in zip(valid_rows, v):
                if not np.array_equal(have, r):
                    raise ValueError('Inconsistent valid rows across subjects')

    ica_train = OrderedDict(
        (s,
         np.concatenate(list(
             r[v] for i, (r, v) in enumerate(zip(fmri_subject_data_partitions[s], valid_rows)) if i != hold_out)))
        for s in fmri_subject_data_partitions)
    if hold_out is None:
        ica_validation = None
    else:
        ica_validation = OrderedDict(
            (s, fmri_subject_data_partitions[s][hold_out][valid_rows[hold_out]])
            for s in fmri_subject_data_partitions)

    ica = FastICA(n_components=num_components, max_iter=max_iter, tol=tol)
    train_sources = ica.fit_transform(np.concatenate(
        list(ica_train[s][:, indicator_train_dict[s]] for s in ica_train), axis=1))
    validation_sources = None
    if ica_validation is not None:
        validation_sources = ica.transform(np.concatenate(
            list(ica_validation[s][:, indicator_train_dict[s]] for s in ica_validation), axis=1))

    sources = list()
    for _ in range(len(valid_rows)):
        sources.append(None)
    train_split = _fill_rows(train_sources, list(v for i, v in enumerate(valid_rows) if i != hold_out))
    for i, t in zip([idx for idx in range(len(valid_rows)) if idx != hold_out], train_split):
        sources[i] = t
    if hold_out is not None:
        sources[hold_out] = validation_sources

    valid_arr_helper = _ValidArrHelper(fmri_subject_data_partitions)

    x = np.concatenate([train_sources, np.ones((len(train_sources), 1), dtype=train_sources.dtype)], axis=1)

    mixing = OrderedDict()
    bias = OrderedDict()

    for s in fmri_subject_data_partitions:
        y = list(valid_arr_helper.compress(d, i, s)
                 for i, d in enumerate(fmri_subject_data_partitions[s]) if i != hold_out)
        solution, _, _, _ = np.linalg.lstsq(x, np.concatenate(y), rcond=None)
        # fill the mixing matrix with nan where y is nan
        filled = np.full((len(solution), len(valid_arr_helper.valid_columns[s])), np.nan)
        filled[:, valid_arr_helper.valid_columns[s]] = solution
        mixing[s] = filled[:-1]
        bias[s] = filled[-1]

    return sources, mixing, bias


def multi_set_kernel_cca_hold_out_projection(
        fmri_subject_data_partitions, hold_out, regularization_candidates, num_components,
        target_fmri_subject_data_predictions=None,
        show_cv_corr=False):

    valid_arr_helper = _ValidArrHelper(fmri_subject_data_partitions)
    target_valid_arr_helper = valid_arr_helper
    if target_fmri_subject_data_predictions is not None:
        target_valid_arr_helper = _ValidArrHelper(target_fmri_subject_data_predictions)

    cca_train = OrderedDict(
        (s,
         list(valid_arr_helper.compress(d, i, s)
              for i, d in enumerate(fmri_subject_data_partitions[s]) if i != hold_out))
        for s in fmri_subject_data_partitions)
    if hold_out is None:
        cca_validation = None
    else:
        cca_validation = OrderedDict(
            (s, valid_arr_helper.compress(fmri_subject_data_partitions[s][hold_out], hold_out, s))
            for s in fmri_subject_data_partitions)

    kcca_result = multi_set_kcca_cv(
        cca_train, regularization_candidates, num_components, cca_validation, show_cv_corr=show_cv_corr)
    if cca_validation is None:
        train_components = kcca_result
        validation_components = None
    else:
        train_components, validation_components = kcca_result

    projected_data = OrderedDict()
    for s in train_components:
        x = np.concatenate(
            [train_components[s], np.ones((len(train_components[s]), 1), train_components[s].dtype)], axis=1)
        y = cca_train[s]
        if target_fmri_subject_data_predictions is not None:
            y = list(target_valid_arr_helper.compress(d, i, s)
                     for i, d in enumerate(target_fmri_subject_data_predictions[s]) if i != hold_out)
        solution, _, _, _ = np.linalg.lstsq(x, np.concatenate(y), rcond=None)
        splits = np.cumsum(list(len(c) for c in y))[:-1]
        projection = np.split(x @ solution, splits)
        projected_data[s] = list()
        for _ in range(target_valid_arr_helper.num_partitions):
            projected_data[s].append(None)

        for i, p in zip([i for i in range(target_valid_arr_helper.num_partitions) if i != hold_out], projection):
            projected_data[s][i] = target_valid_arr_helper.fill(p, i, s)

        if hold_out is not None:
            x = np.concatenate(
                [validation_components[s],
                 np.ones((len(validation_components[s]), 1), validation_components[s].dtype)], axis=1)
            y = cca_validation[s]
            if target_fmri_subject_data_predictions is not None:
                y = target_valid_arr_helper.compress(target_fmri_subject_data_predictions[s][hold_out], hold_out, s)
            solution, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
            projected_data[s][hold_out] = target_valid_arr_helper.fill(x @ solution, hold_out, s)
    return projected_data


def one_from_all_kernel_ridge_cv_correlation(
        fmri_subject_data_partitions,
        alpha_candidates,
        mid_range_warn_level=0.05,
        use_cca_projection=False,
        cca_regularization_candidates=1 - np.linspace(1e-7, 1e-5, 20),
        cca_num_components=None):
    num_partitions = None
    for s in fmri_subject_data_partitions:
        if num_partitions is None:
            num_partitions = len(fmri_subject_data_partitions[s])
        elif num_partitions != len(fmri_subject_data_partitions[s]):
            raise ValueError('Data partitions inconsistent across subjects')

    if num_partitions < 3:
        raise ValueError('There must be at least 3 partitions of the data to run this correlation')

    valid_arr_helper = _ValidArrHelper(fmri_subject_data_partitions)

    all_predictions = list()
    full_result = dict()

    progress = tqdm(total=num_partitions * len(fmri_subject_data_partitions))

    validation_data = OrderedDict()
    projected_data = list()
    for hold_out in range(num_partitions):
        all_predictions.append(dict())

        if use_cca_projection:
            if cca_regularization_candidates is None:
                raise ValueError('cca_regularization_candidates must be specified if using cca_projection')
            data_partitions = multi_set_kernel_cca_hold_out_projection(
                fmri_subject_data_partitions, hold_out, cca_regularization_candidates, cca_num_components)
            projected_data.append(data_partitions)
        else:
            data_partitions = fmri_subject_data_partitions

        for subject in _progress_iterate(progress, data_partitions):
            other = list()
            self = list()
            for index_partition in range(num_partitions):
                if index_partition == hold_out:
                    continue
                other.append(list())
                for s2 in data_partitions:
                    if s2 == subject:
                        continue
                    other[-1].append(valid_arr_helper.compress(
                        data_partitions[s2][index_partition], index_partition, s2))
                other[-1] = np.concatenate(other[-1], axis=1)
                self_partition = valid_arr_helper.compress(
                    data_partitions[subject][index_partition], index_partition, subject)
                self.append(self_partition)
            other_validation = np.concatenate(list(
                valid_arr_helper.compress(data_partitions[s2][hold_out], hold_out, s2)
                for s2 in data_partitions if s2 != subject), axis=1)

            if subject not in validation_data:
                validation_data[subject] = list()

            validation_data[subject].append(data_partitions[subject][hold_out])

            self_validation = valid_arr_helper.compress(
                data_partitions[subject][hold_out], hold_out, subject)

            mse, y_hat, selected_alpha = kernel_ridge_cv(
                other,
                self,
                other_validation,
                self_validation,
                alpha_candidates)

            indicator_less_max = np.max(alpha_candidates) > selected_alpha
            indicator_greater_min = selected_alpha > np.min(alpha_candidates)
            num_mid_range = np.count_nonzero(np.logical_and(indicator_less_max, indicator_greater_min))
            if num_mid_range / np.size(selected_alpha) < mid_range_warn_level:
                print('Warning: only {} of selected alpha values are in the middle of the available values '
                      '({} less than max, {} greater than min). Consider modifying alpha_candidates'.format(
                            num_mid_range,
                            np.count_nonzero(indicator_less_max),
                            np.count_nonzero(indicator_greater_min)))

            del other
            del self

            # fill the invalid columns/rows
            y_hat = valid_arr_helper.fill(y_hat, hold_out, subject)
            selected_alpha = valid_arr_helper.fill(selected_alpha, hold_out, subject)

            full_result[(subject, hold_out)] = mse, y_hat, selected_alpha
            all_predictions[-1][subject] = y_hat

    progress.close()

    correlations = dict()
    for hold_out in range(len(all_predictions) + 1):
        if hold_out == len(all_predictions):
            for subject in all_predictions[0]:
                y_hat = list()
                y = list()
                for i in range(len(all_predictions)):
                    y_hat.append(all_predictions[i][subject])
                    y_partition = validation_data[subject][i]
                    y.append(y_partition)
                y_hat = np.concatenate(y_hat)
                y = np.concatenate(y)
                correlations[subject] = np.nanmean(eps_standardize(y) * eps_standardize(y_hat), axis=0)
        else:
            for subject in all_predictions[hold_out]:
                y_hat = all_predictions[hold_out][subject]
                y = validation_data[subject][hold_out]
                correlations[(subject, hold_out)] = np.nanmean(eps_standardize(y) * eps_standardize(y_hat), axis=0)

    correlations = OrderedDict((k, correlations[k]) for k in sorted(correlations, key=_tuple_key))
    full_result = OrderedDict((k, full_result[k]) for k in sorted(full_result, key=_tuple_key))

    if use_cca_projection:
        return correlations, full_result, projected_data

    return correlations, full_result


def means_of_clusters(clusters, data):
    data = data[:, clusters >= 0]
    clusters = clusters[clusters >= 0]
    num_clusters = np.max(clusters) + 1
    # compute the means within cluster for all data (including test data)
    segments = np.reshape(
        np.expand_dims(np.arange(len(data)), 1) * num_clusters + np.expand_dims(clusters, 0), -1)

    clustered_data = np.bincount(segments, weights=np.reshape(data, -1)) / np.bincount(segments)
    return np.reshape(clustered_data, (len(data), num_clusters))


def cluster_predictable_voxels(
        correlations,
        data,
        num_clusters=12,
        num_init=10,
        correlation_threshold=0.1,
        use_ranked_data=True,
        include_no_hold_out=False,
        which_hold_out=None):
    grouped_keys = dict()
    for key in correlations:
        if isinstance(key, tuple):
            _, hold_out = key
        else:
            hold_out = -1
        if hold_out < 0 and not include_no_hold_out:
            continue
        if which_hold_out is not None and hold_out != which_hold_out:
            continue
        if hold_out not in grouped_keys:
            grouped_keys[hold_out] = list()
        grouped_keys[hold_out].append(key)

    output_data = dict()
    output_clusters = dict()
    output_means = dict()
    for hold_out in tqdm(sorted(grouped_keys)):
        include_indicators = OrderedDict()
        joint_train_data = list()
        joint_data = list()
        valid_rows = None
        for key in grouped_keys[hold_out]:
            if isinstance(key, tuple):
                subject, _ = key
            else:
                subject = key
            if subject in include_indicators:
                raise ValueError('Bad correlations dictionary. Multiple entries for a subject within one hold out')
            train_data = np.concatenate(
                list(r for i, r in enumerate(data[subject]) if i != hold_out))
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                include_indicators[subject] = correlations[key] >= correlation_threshold

            t = train_data[:, include_indicators[subject]]
            if use_ranked_data:
                t = nanrankdata(t, axis=0)
            mu = np.mean(t, axis=1, keepdims=True)
            t = t - mu
            if use_ranked_data:
                t = nanrankdata(t, axis=0)
            is_finite = np.isfinite(t)
            subject_bad_rows = np.any(np.logical_not(is_finite), axis=1)
            if np.any(np.logical_and(subject_bad_rows, np.any(is_finite, axis=1))):
                raise ValueError('Rows must have all finite values or all nan values')

            if valid_rows is None:
                valid_rows = np.logical_not(subject_bad_rows)
            elif not np.array_equal(valid_rows, np.logical_not(subject_bad_rows)):
                raise ValueError('All subjects must have the same nan rows')
            joint_train_data.append(t[valid_rows])
            d = np.concatenate(data[subject])[:, include_indicators[subject]]
            if use_ranked_data:
                d = nanrankdata(d, axis=0)
            mu = np.mean(d, axis=1)
            d = d - np.expand_dims(mu, 1)
            if use_ranked_data:
                d = nanrankdata(d, axis=0)
            joint_data.append(d)
            output_means[(subject, hold_out)] = mu
        joint_train_data = np.concatenate(joint_train_data, axis=1)
        joint_data = np.concatenate(joint_data, axis=1)

        # cluster
        k_means = KMeans(n_clusters=num_clusters, n_init=num_init)
        clusters = k_means.fit_predict(joint_train_data.T)

        # create cluster assignment vectors
        offset = 0
        full_clusters = OrderedDict()
        for subject in include_indicators:
            subject_clusters = -1 * np.ones(include_indicators[subject].shape, dtype=np.int32)
            c = clusters[offset:np.sum(include_indicators[subject]) + offset]
            subject_clusters[include_indicators[subject]] = c
            offset += np.sum(include_indicators[subject])
            full_clusters[subject] = subject_clusters

        clustered_data = means_of_clusters(clusters, joint_data)

        if hold_out - 1 in output_data:
            # try to number the clusters consistently across folds (just for better visualization)
            last_clustered_data = output_data[hold_out - 1]
            valid_rows = np.logical_and(
                np.all(np.isfinite(last_clustered_data), axis=1), np.all(np.isfinite(clustered_data), axis=1))
            distances = cdist(clustered_data[valid_rows].T, last_clustered_data[valid_rows].T)
            _, new_numbers = linear_sum_assignment(distances)
            for subject in full_clusters:
                # replace the cluster assignments with the new numbers
                full_clusters[subject] = np.where(
                    full_clusters[subject] >= 0,
                    new_numbers[np.where(full_clusters[subject] >= 0, full_clusters[subject], 0)],
                    full_clusters[subject])
            clustered_data = clustered_data[:, np.argsort(new_numbers)]

        for subject in full_clusters:
            output_clusters[(subject, hold_out)] = full_clusters[subject]
        output_data[hold_out] = clustered_data

    output_data = OrderedDict((k, output_data[k]) for k in sorted(output_data))
    output_clusters = OrderedDict((k, output_clusters[k]) for k in sorted(output_clusters))
    output_means = OrderedDict((k, output_means[k]) for k in sorted(output_means))

    return output_clusters, output_data, output_means


def _is_local_maxima(x, order=1):
    indicator = np.full(x.shape, True)
    offsets_1d = [np.arange(-order, order + 1)] * len(x.shape)
    mesh = np.meshgrid(*offsets_1d)
    offsets = np.concatenate(list(np.reshape(m, (-1, 1)) for m in mesh), axis=1)
    offsets = offsets[np.logical_not(np.all(offsets == 0, axis=1))]

    indices = list(np.arange(x.shape[axis]) for axis in range(np.ndim(x)))

    for offset in offsets:
        shifted = x
        for axis, (ind, s) in enumerate(zip(indices, offset)):
            shifted = np.take(shifted, ind + s, mode='clip', axis=axis)
        indicator = np.logical_and(indicator, x >= shifted)

    return indicator


def is_local_maxima_from_value_dict(value_dict, masks, min_value=None, order=1):
    indicator_dict = type(value_dict)()
    for key in value_dict:
        if isinstance(key, tuple):
            subject, hold_out = key
        else:
            subject = key

        v = np.full(masks[subject].shape, -np.inf)
        v[masks[subject]] = value_dict[key]
        indicator_dict[key] = _is_local_maxima(v, order=order)[masks[subject]]

        if min_value is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                indicator_dict[key] = np.logical_and(indicator_dict[key], value_dict[key] >= min_value)

    return indicator_dict


def joint_cluster_local_maxima(
        indicator_local_maxima_dict, data, correlations, correlation_threshold,
        num_clusters, num_init, which_hold_out=None):
    grouped_keys = dict()
    for key in indicator_local_maxima_dict:
        if isinstance(key, tuple):
            _, hold_out = key
        else:
            hold_out = -1
        if which_hold_out is not None and hold_out != which_hold_out:
            continue
        if hold_out not in grouped_keys:
            grouped_keys[hold_out] = list()
        grouped_keys[hold_out].append(key)

    output_clusters = dict()
    output_data = dict()
    for hold_out in grouped_keys:
        valid_rows = None
        joint_train_data = list()
        for key in grouped_keys[hold_out]:
            if isinstance(key, tuple):
                subject, _ = key
            else:
                subject = key
            indicator_local_maxima = indicator_local_maxima_dict[key]
            train_data = np.concatenate(
                list(r for i, r in enumerate(data[subject]) if i != hold_out))
            train_data = train_data[:, indicator_local_maxima]
            is_finite = np.isfinite(train_data)
            subject_bad_rows = np.any(np.logical_not(is_finite), axis=1)
            if np.any(np.logical_and(subject_bad_rows, np.any(is_finite, axis=1))):
                raise ValueError('Rows must have all finite values or all nan values')
            if valid_rows is None:
                valid_rows = np.logical_not(subject_bad_rows)
            elif not np.array_equal(valid_rows, np.logical_not(subject_bad_rows)):
                raise ValueError('All subjects must have the same nan rows')

            train_data = train_data[valid_rows]
            train_data = eps_standardize(train_data)

            joint_train_data.append(train_data)

        num_maxima = sum(t.shape[1] for t in joint_train_data)
        self_affinity = np.zeros((num_maxima, num_maxima), dtype=joint_train_data[0].dtype)
        offset = 0
        for t in joint_train_data:
            self_affinity[offset:offset + t.shape[1], offset:offset + t.shape[1]] = 1 + np.corrcoef(t, rowvar=False)
            offset += t.shape[1]
        joint_train_data = np.concatenate(joint_train_data, axis=1)
        affinity = 1 + np.corrcoef(joint_train_data, rowvar=False)
        affinity = affinity - self_affinity

        spectral = SpectralClustering(n_clusters=num_clusters, n_init=num_init, affinity='precomputed')
        labels = spectral.fit_predict(affinity)

        # compute the mean within label
        label_means = means_of_clusters(labels, joint_train_data)
        output_data[hold_out] = label_means

        # assign the voxels to a label
        for key in grouped_keys[hold_out]:
            if isinstance(key, tuple):
                subject, _ = key
            else:
                subject = key
            train_data = np.concatenate(
                list(r for i, r in enumerate(data[subject]) if i != hold_out))
            is_finite = np.isfinite(train_data)
            subject_bad_rows = np.any(np.logical_not(is_finite), axis=1)
            if np.any(np.logical_and(subject_bad_rows, np.any(is_finite, axis=1))):
                raise ValueError('Rows must have all finite values or all nan values')
            if valid_rows is None:
                valid_rows = np.logical_not(subject_bad_rows)
            elif not np.array_equal(valid_rows, np.logical_not(subject_bad_rows)):
                raise ValueError('All subjects must have the same nan rows')

            train_data = train_data[valid_rows]
            train_data = eps_standardize(train_data)

            labels_key = np.argmin(cdist(
                train_data[:, correlations[key] >= correlation_threshold].T,
                label_means.T), axis=1)

            output_clusters[key] = np.full(correlations[key].shape, -1, dtype=labels.dtype)
            output_clusters[key][correlations[key] >= correlation_threshold] = labels_key

    output_data = OrderedDict((k, output_data[k]) for k in sorted(output_data))
    output_clusters = OrderedDict((k, output_clusters[k]) for k in sorted(output_clusters))

    return output_clusters, output_data


def ordered_gram_schmidt(x, sort_values):
    if len(sort_values) != x.shape[1]:
        raise ValueError('Expected len(sort_values) to match x.shape[1]')
    sorter = np.argsort(sort_values)
    inverse = np.argsort(sorter)
    return modified_gram_schmidt(x[:, sorter])[:, inverse]


def gram_schmidt_joint_cluster_local_maxima(
        indicator_local_maxima_dict, data, correlations, correlation_threshold,
        num_clusters, num_init, which_hold_out=None):
    grouped_keys = dict()
    for key in indicator_local_maxima_dict:
        if isinstance(key, tuple):
            _, hold_out = key
        else:
            hold_out = -1
        if which_hold_out is not None and hold_out != which_hold_out:
            continue
        if hold_out not in grouped_keys:
            grouped_keys[hold_out] = list()
        grouped_keys[hold_out].append(key)

    output_clusters = dict()
    output_data = dict()
    for hold_out in grouped_keys:
        valid_rows = None
        joint_train_data = list()
        joint_train_data_correlations = list()
        non_zero_maxima = list()
        for key in grouped_keys[hold_out]:
            if isinstance(key, tuple):
                subject, _ = key
            else:
                subject = key
            indicator_local_maxima = indicator_local_maxima_dict[key]
            indicator_local_maxima = np.logical_and(indicator_local_maxima, correlations[key] >= correlation_threshold)
            train_data = np.concatenate(
                list(r for i, r in enumerate(data[subject]) if i != hold_out))
            train_data = train_data[:, indicator_local_maxima]
            is_finite = np.isfinite(train_data)
            subject_bad_rows = np.any(np.logical_not(is_finite), axis=1)
            if np.any(np.logical_and(subject_bad_rows, np.any(is_finite, axis=1))):
                raise ValueError('Rows must have all finite values or all nan values')
            if valid_rows is None:
                valid_rows = np.logical_not(subject_bad_rows)
            elif not np.array_equal(valid_rows, np.logical_not(subject_bad_rows)):
                raise ValueError('All subjects must have the same nan rows')

            train_data = train_data[valid_rows]
            train_data_gs = ordered_gram_schmidt(train_data, -correlations[key][indicator_local_maxima])
            non_zero = np.sum(train_data_gs, axis=0) != 0
            train_data = train_data[:, non_zero]
            train_data_gs = train_data_gs[:, non_zero]
            joint_train_data.append(train_data_gs)
            non_zero_maxima.append(train_data)
            joint_train_data_correlations.append(correlations[key][indicator_local_maxima][non_zero])

        joint_train_data = np.concatenate(joint_train_data, axis=1)
        joint_train_data_correlations = np.concatenate(joint_train_data_correlations, axis=1)
        non_zero_maxima = np.concatenate(non_zero_maxima, axis=1)
        affinity = joint_train_data.T @ joint_train_data

        spectral = SpectralClustering(n_clusters=num_clusters, n_init=num_init, affinity='precomputed')
        labels = spectral.fit_predict(affinity)

        # compute the mean within label
        label_means = means_of_clusters(labels, joint_train_data)
        label_mean_correlation = np.bincount(labels, weights=joint_train_data_correlations / np.bincount(labels))

        # re-orthogonalize
        label_means = ordered_gram_schmidt(label_means, -label_mean_correlation)
        output_data[hold_out] = label_means

        # find the dot-product of the original maxima with clusters
        maxima_dot = label_means.T @ non_zero_maxima

        # assign the voxels to a label
        for key in grouped_keys[hold_out]:
            if isinstance(key, tuple):
                subject, _ = key
            else:
                subject = key
            train_data = np.concatenate(
                list(r for i, r in enumerate(data[subject]) if i != hold_out))
            is_finite = np.isfinite(train_data)
            subject_bad_rows = np.any(np.logical_not(is_finite), axis=1)
            if np.any(np.logical_and(subject_bad_rows, np.any(is_finite, axis=1))):
                raise ValueError('Rows must have all finite values or all nan values')
            if valid_rows is None:
                valid_rows = np.logical_not(subject_bad_rows)
            elif not np.array_equal(valid_rows, np.logical_not(subject_bad_rows)):
                raise ValueError('All subjects must have the same nan rows')

            train_data = train_data[valid_rows]
            train_dot = label_means.T @ train_data

            # find the nearest neighbor in the dot product space
            labels_key = labels[np.argmin(cdist(train_dot.T, maxima_dot.T), axis=1)]
            output_clusters[key] = np.where(correlations[key] >= correlation_threshold, labels_key, -1)

    output_data = OrderedDict((k, output_data[k]) for k in sorted(output_data))
    output_clusters = OrderedDict((k, output_clusters[k]) for k in sorted(output_clusters))

    return output_clusters, output_data


def _nearest(x, k):
    indices_sort = np.argsort(-x, axis=1)
    indices_sort = indices_sort[:, :k]
    indices_sort_b = np.tile(
        np.expand_dims(np.arange(indices_sort.shape[0]), 1), (1, indices_sort.shape[1]))
    indices_sort = np.reshape(indices_sort, -1)
    indices_sort_b = np.reshape(indices_sort_b, -1)
    indicator_nearest = np.full_like(x, False)
    indicator_nearest[(indices_sort, indices_sort_b)] = True
    indicator_nearest[(indices_sort_b, indices_sort)] = True
    return indicator_nearest


def _valid_rows(x):
    is_finite = np.isfinite(x)
    bad_rows = np.any(np.logical_not(is_finite), axis=1)
    if np.any(np.logical_and(bad_rows, np.any(is_finite, axis=1))):
        raise ValueError('Rows must have all finite values or all nan values')
    return np.logical_not(bad_rows)


def _fill_rows(x, valid_rows, split=True, fill_value=np.nan):
    if split:
        splits = np.cumsum(list(np.sum(v) for v in valid_rows))[:-1]
        x = np.split(x, splits)
    else:
        x = [x]
        valid_rows = [valid_rows]
    result = list()
    for x_, v in zip(x, valid_rows):
        result.append(np.full((len(v),) + x_.shape[1:], fill_value=fill_value, dtype=x_.dtype))
        result[-1][v] = x_
    if not split:
        result = result[0]
    return result


def residual_directions(
        indicator_local_maxima_dict,
        data,
        num_pca_components,
        num_spectral_components,
        num_spectral_init=10,
        num_iter=3,
        which_hold_out=None):
    grouped_keys = dict()
    for key in indicator_local_maxima_dict:
        if isinstance(key, tuple):
            _, hold_out = key
        else:
            hold_out = -1
        if which_hold_out is not None and hold_out != which_hold_out:
            continue
        if hold_out not in grouped_keys:
            grouped_keys[hold_out] = list()
        grouped_keys[hold_out].append(key)

    result = dict()
    for hold_out in grouped_keys:
        for key in grouped_keys[hold_out]:
            if isinstance(key, tuple):
                subject, _ = key
            else:
                subject = key

            valid_rows = list(_valid_rows(r) for r in data[subject])
            compressed = list(r[v] for r, v in zip(data[subject], valid_rows))
            train = np.concatenate(list(r for i, r in enumerate(compressed) if i != hold_out))
            all_data = np.concatenate(compressed)

            maxima = train[:, indicator_local_maxima_dict[key]]
            full_maxima = all_data[:, indicator_local_maxima_dict[key]]
            labels = np.arange(maxima.shape[1])
            maxima_residual = None
            full_maxima_residual = None
            for _ in range(num_iter):
                maxima_residual = np.full_like(maxima, np.nan)
                full_maxima_residual = np.full_like(full_maxima, np.nan)
                for i in np.unique(labels):
                    indicator_x = labels != i
                    indicator_y = labels == i

                    pca = PCA(n_components=num_pca_components)
                    other = pca.fit_transform(maxima[:, indicator_x])

                    other = np.concatenate([other, np.ones((len(other), 1), dtype=other.dtype)], axis=1)
                    solution, _, _, _ = np.linalg.lstsq(other, maxima[:, indicator_y], rcond=None)
                    maxima_residual[:, indicator_y] = maxima[:, indicator_y] - other @ solution

                    other_full = pca.transform(full_maxima[:, indicator_x])
                    other_full = np.concatenate(
                        [other_full, np.ones((len(other_full), 1), dtype=other_full.dtype)], axis=1)
                    full_maxima_residual[:, indicator_y] = full_maxima[:, indicator_y] - other_full @ solution

                affinity = 1 - squareform(pdist(maxima_residual.T, metric='cosine'))
                indicator_nearest = _nearest(affinity, num_spectral_components)
                affinity = np.where(indicator_nearest, affinity, 0)
                spectral = SpectralClustering(
                    n_clusters=num_spectral_components, n_init=num_spectral_init, affinity='precomputed')
                labels = spectral.fit_predict(affinity)

            # choose a centroid
            indicator_centroids = np.full(np.sum(indicator_local_maxima_dict[key]), False)
            for i in np.unique(labels):
                indicator_label = labels == i
                affinity = 1 - squareform(pdist(maxima_residual[:, indicator_label].T, metric='cosine'))
                indicator_max = np.full(len(affinity), False)
                indicator_max[np.argmax(np.sum(affinity, axis=1))] = True
                indicator_centroids[indicator_label] = indicator_max

            train_centroids = maxima[:, indicator_centroids]
            full_centroids = full_maxima_residual[:, indicator_centroids]

            full_centroid_residuals = np.full_like(full_centroids, np.nan)
            for i in range(train_centroids.shape[1]):
                indicator_x = np.arange(train_centroids.shape[1]) != i
                other = np.concatenate(
                    [train_centroids[:, indicator_x], np.ones((len(train_centroids), 1), dtype=train_centroids.dtype)],
                    axis=1)
                solution, _, _, _ = np.linalg.lstsq(other, train_centroids[:, i], rcond=None)
                other = np.concatenate(
                    [full_centroids[:, indicator_x], np.ones((len(full_centroids), 1), dtype=full_centroids.dtype)],
                    axis=1)
                full_centroid_residuals[:, i] = full_centroids[:, i] - other @ solution

            result[key] = _fill_rows(full_centroid_residuals, valid_rows)

    result = OrderedDict((k, result[k]) for k in sorted(result))
    return result


def joint_cluster_maxima_residual(maxima_residuals, data, num_components, which_hold_out=None, num_spectral_init=10):
    grouped_keys = dict()
    for key in maxima_residuals:
        if isinstance(key, tuple):
            _, hold_out = key
        else:
            hold_out = -1
        if which_hold_out is not None and hold_out != which_hold_out:
            continue
        if hold_out not in grouped_keys:
            grouped_keys[hold_out] = list()
        grouped_keys[hold_out].append(key)

    result = dict()
    components = dict()
    for hold_out in grouped_keys:

        valid_rows = None
        joint_train_data = list()
        joint_train_maxima_residuals = list()
        joint_full_maxima_residuals = list()

        for key in grouped_keys[hold_out]:
            if isinstance(key, tuple):
                subject, _ = key
            else:
                subject = key

            v = list(_valid_rows(r) for r in data[subject])
            if valid_rows is None:
                valid_rows = v
            else:
                if len(valid_rows) != len(v):
                    raise ValueError('All subjects must have the same nan rows')
                for v_, have in zip(v, valid_rows):
                    if not np.array_equal(v_, have):
                        raise ValueError('All subjects must have the same nan rows')

            compressed = list(r[v] for r, v in zip(data[subject], valid_rows))
            train = np.concatenate(list(r for i, r in enumerate(compressed) if i != hold_out))

            full_maxima = list(r[v] for r, v in zip(maxima_residuals[key], valid_rows))
            train_maxima = np.concatenate(list(r for i, r in enumerate(full_maxima) if i != hold_out))

            joint_train_data.append(train)
            joint_train_maxima_residuals.append(train_maxima)
            joint_full_maxima_residuals.append(np.concatenate(full_maxima))

        maxima_counts = list(m.shape[1] for m in joint_train_maxima_residuals)
        joint_train_maxima_residuals = np.concatenate(joint_train_maxima_residuals, axis=1)
        joint_full_maxima_residuals = np.concatenate(joint_full_maxima_residuals, axis=1)

        affinity = 1 - squareform(pdist(joint_train_maxima_residuals.T, metric='cosine'))
        # set block diagonal to 0
        offset = 0
        for m in maxima_counts:
            affinity[offset:offset + m, offset:offset + m] = 0
            offset += m

        indicator_nearest = _nearest(affinity, num_components)
        affinity = np.where(indicator_nearest, affinity, 0)

        spectral = SpectralClustering(n_clusters=num_components, n_init=num_spectral_init, affinity='precomputed')
        labels = spectral.fit_predict(affinity)

        train_components = means_of_clusters(labels, joint_train_maxima_residuals)
        full_components = means_of_clusters(labels, joint_full_maxima_residuals)
        full_component_residuals = np.full_like(full_components, np.nan)
        train_component_residuals = np.full_like(train_components, np.nan)
        for i in range(train_components.shape[1]):
            indicator_x = np.arange(train_components.shape[1]) != i
            other = np.concatenate(
                [train_components[:, indicator_x], np.ones((len(train_components), 1), dtype=train_components.dtype)],
                axis=1)
            solution, _, _, _ = np.linalg.lstsq(other, train_components[:, i], rcond=None)
            train_component_residuals[:, i] = train_components[:, i] - other @ solution
            other = np.concatenate(
                [full_components[:, indicator_x], np.ones((len(full_components), 1), dtype=full_components.dtype)],
                axis=1)
            full_component_residuals[:, i] = full_components[:, i] - other @ solution

        train_components = train_component_residuals
        full_components = full_component_residuals

        if hold_out - 1 in components:
            # try to number the clusters consistently across folds (just for better visualization)
            last_clustered_data = components[hold_out - 1]
            last_clustered_data = np.concatenate(list(r[v] for r, v in zip(last_clustered_data, valid_rows)))
            distances = cdist(full_components.T, last_clustered_data.T, metric='cosine')
            _, new_numbers = linear_sum_assignment(distances)
            sorter = np.argsort(new_numbers)
            full_components = full_components[:, sorter]
            train_components = train_components[:, sorter]

        components[hold_out] = _fill_rows(full_components, valid_rows)

        for key, train in zip(grouped_keys[hold_out], joint_train_data):
            result[key] = labels[np.argmin(cdist(train.T, train_components.T, metric='cosine'), axis=1)]

    result = OrderedDict((k, result[k]) for k in maxima_residuals)
    components = OrderedDict((k, components[k]) for k in sorted(components))
    return components, result
