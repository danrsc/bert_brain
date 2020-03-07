from collections import OrderedDict
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from tqdm.auto import tqdm
from bottleneck import nanrankdata

from kernel_ridge_cv_module import kernel_ridge_cv


def eps_standardize(x, axis=0, eps=1e-12):
    std = np.nanstd(x, axis=axis, keepdims=True)
    return np.where(std < eps, np.nan, np.divide(x - np.nanmean(x, axis=axis, keepdims=True), std, where=std >= eps))


def _tuple_key(k):
    if isinstance(k, str):
        return k,
    return k


def pair_predict_ensemble_kernel_ridge_cv_correlation(
        fmri_subject_data_partitions, alpha_candidates, max_workers=None):
    num_partitions = None
    for s in fmri_subject_data_partitions:
        if num_partitions is None:
            num_partitions = len(fmri_subject_data_partitions[s])
        elif num_partitions != len(fmri_subject_data_partitions[s]):
            raise ValueError('Data partitions inconsistent across subjects')

    all_predictions = list()
    full_result = dict()
    for hold_out in range(num_partitions):
        print('starting hold out {}'.format(hold_out))
        all_predictions.append(dict())

        train_data = dict((s, list(r for i, r in enumerate(fmri_subject_data_partitions[s]) if i != hold_out))
                          for s in fmri_subject_data_partitions)

        pairs = list()
        for s1 in fmri_subject_data_partitions:
            for s2 in fmri_subject_data_partitions:
                if s1 != s2:
                    pairs.append((s1, s2))

        if max_workers is not None and max_workers < 2:
            for result, (s1, s2) in tqdm(zip(map(
                    kernel_ridge_cv,
                    [train_data[s1] for s1, s2 in pairs],
                    [train_data[s2] for s1, s2 in pairs],
                    [fmri_subject_data_partitions[s1][hold_out] for s1, s2 in pairs],
                    [fmri_subject_data_partitions[s2][hold_out] for s1, s2 in pairs],
                    [alpha_candidates for _ in pairs],
                    [-1 for _ in pairs]), pairs), total=len(pairs)):
                full_result[(s1, s2, hold_out)] = result
                prediction = full_result[(s1, s2, hold_out)][1]
                if s2 not in all_predictions[-1]:
                    all_predictions[-1][s2] = list()
                all_predictions[-1][s2].append(np.expand_dims(prediction, 0))
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                for result, (s1, s2) in tqdm(zip(ex.map(
                        kernel_ridge_cv,
                        [train_data[s1] for s1, s2 in pairs],
                        [train_data[s2] for s1, s2 in pairs],
                        [fmri_subject_data_partitions[s1][hold_out] for s1, s2 in pairs],
                        [fmri_subject_data_partitions[s2][hold_out] for s1, s2 in pairs],
                        [alpha_candidates for _ in pairs],
                        [-1 for _ in pairs]), pairs), total=len(pairs)):
                    full_result[(s1, s2, hold_out)] = result
                    prediction = full_result[(s1, s2, hold_out)][1]
                    if s2 not in all_predictions[-1]:
                        all_predictions[-1][s2] = list()
                    all_predictions[-1][s2].append(np.expand_dims(prediction, 0))

        for subject in all_predictions[-1]:
            # take the mean prediction over the other subjects
            all_predictions[-1][subject] = np.mean(np.concatenate(all_predictions[-1][subject]), axis=0)

    correlations = dict()
    for hold_out in range(len(all_predictions) + 1):
        if hold_out == len(all_predictions):
            for subject in all_predictions[0]:
                y_hat = np.concatenate(list(all_predictions[i][subject] for i in range(len(all_predictions))))
                y = np.concatenate(fmri_subject_data_partitions[subject])
                correlations[subject] = np.nanmean(eps_standardize(y) * eps_standardize(y_hat), axis=0)
        else:
            for subject in all_predictions[hold_out]:
                y_hat = all_predictions[hold_out][subject]
                y = fmri_subject_data_partitions[subject][hold_out]
                correlations[(subject, hold_out)] = np.nanmean(eps_standardize(y) * eps_standardize(y_hat), axis=0)

    correlations = OrderedDict((k, correlations[k]) for k in sorted(correlations, key=_tuple_key))
    full_result = OrderedDict((k, full_result[k]) for k in sorted(full_result, key=_tuple_key))

    return correlations, full_result


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
                subject_bad_rows = np.any(np.logical_not(is_finite), axis=1)
                if np.any(np.logical_and(subject_bad_rows, np.any(is_finite, axis=1))):
                    raise ValueError('Rows must have all finite values or all nan values')
                if index_partition not in valid_rows:
                    valid_rows[index_partition] = np.logical_not(subject_bad_rows)
                elif not np.array_equal(valid_rows[index_partition], np.logical_not(subject_bad_rows)):
                    raise ValueError('All subjects must have the same nan rows')
                subject_bad_columns = np.any(np.logical_not(is_finite[valid_rows[index_partition]]), axis=0)
                if np.any(np.logical_and(subject_bad_columns, np.any(is_finite[valid_rows[index_partition]], axis=0))):
                    raise ValueError('Columns must have all finite values or all nan values')
                if subject not in valid_columns:
                    valid_columns[subject] = np.logical_not(subject_bad_columns)
                else:
                    valid_columns[subject] = np.logical_and(valid_columns[subject], np.logical_not(subject_bad_columns))
        self.valid_rows = valid_rows
        self.valid_columns = valid_columns

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


def one_from_all_kernel_ridge_cv_correlation(fmri_subject_data_partitions, alpha_candidates, mid_range_warn_level=0.05):
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

    for hold_out in range(num_partitions):
        all_predictions.append(dict())

        for subject in _progress_iterate(progress, fmri_subject_data_partitions):
            other = list()
            self = list()
            for index_partition in range(num_partitions):
                if index_partition == hold_out:
                    continue
                other.append(list())
                for s2 in fmri_subject_data_partitions:
                    if s2 == subject:
                        continue
                    other[-1].append(valid_arr_helper.compress(
                        fmri_subject_data_partitions[s2][index_partition], index_partition, s2))
                other[-1] = np.concatenate(other[-1], axis=1)
                self.append(valid_arr_helper.compress(
                    fmri_subject_data_partitions[subject][index_partition], index_partition, subject))
            other_validation = np.concatenate(list(
                valid_arr_helper.compress(fmri_subject_data_partitions[s2][hold_out], hold_out, s2)
                for s2 in fmri_subject_data_partitions if s2 != subject), axis=1)
            mse, y_hat, selected_alpha = kernel_ridge_cv(
                other,
                self,
                other_validation,
                valid_arr_helper.compress(fmri_subject_data_partitions[subject][hold_out], hold_out, subject),
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
                y_hat = np.concatenate(list(all_predictions[i][subject] for i in range(len(all_predictions))))
                y = np.concatenate(fmri_subject_data_partitions[subject])
                correlations[subject] = np.nanmean(eps_standardize(y) * eps_standardize(y_hat), axis=0)
        else:
            for subject in all_predictions[hold_out]:
                y_hat = all_predictions[hold_out][subject]
                y = fmri_subject_data_partitions[subject][hold_out]
                correlations[(subject, hold_out)] = np.nanmean(eps_standardize(y) * eps_standardize(y_hat), axis=0)

    correlations = OrderedDict((k, correlations[k]) for k in sorted(correlations, key=_tuple_key))
    full_result = OrderedDict((k, full_result[k]) for k in sorted(full_result, key=_tuple_key))

    return correlations, full_result


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

        # compute the means within cluster for all data (including test data)
        segments = np.reshape(
            np.expand_dims(np.arange(len(joint_data)), 1) * num_clusters + np.expand_dims(clusters, 0), -1)

        clustered_data = np.bincount(segments, weights=np.reshape(joint_data, -1)) / np.bincount(segments)
        clustered_data = np.reshape(clustered_data, (len(joint_data), num_clusters))

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
