import os
import json
import numpy as np
from run_regression import task_hash


def print_variation_results(paths, variation_set_name, training_variation, aux_loss, num_runs):
    output_dir = os.path.join(paths.base_path, 'bert', variation_set_name, task_hash(set(training_variation)))
    aggregated = dict()
    for index_run in range(num_runs):
        run_aggregated = dict()
        validation_json_path = os.path.join(output_dir, 'run_{}'.format(index_run), 'output_validation.json')
        with open(validation_json_path, 'rt') as validation_json_file:
            output_results = json.load(validation_json_file)
        for output_result in output_results:
            name = output_result['name']
            if name not in training_variation and name not in aux_loss:
                continue
            if name not in run_aggregated:
                run_aggregated[name] = (list(), list(), list())
            run_aggregated[name][0].extend(output_result['prediction'])
            run_aggregated[name][1].extend(output_result['target'])
            run_aggregated[name][2].extend(output_result['mask'])
        for name in run_aggregated:
            predictions = np.array(run_aggregated[name][0])
            target = np.array(run_aggregated[name][1])
            mask = np.array(run_aggregated[name][2])
            masked_target = np.where(mask, target, np.nan)
            variance = np.nanvar(masked_target)
            mse = np.nanmean(np.square(predictions - masked_target))
            povu = mse / variance
            pove = 1 - povu
            if name not in aggregated:
                aggregated[name] = (list(), list(), list(), list())
            aggregated[name][0].append(mse)
            aggregated[name][1].append(pove)
            aggregated[name][2].append(povu)
            aggregated[name][3].append(variance)

    print('Variation: {}'.format(', '.join(sorted(training_variation))))
    print('{name:8}  {mse:>10}  {pove:>10}  {povu:>10}  {variance:>10}'.format(
        name='name', mse='mse', pove='pove', povu='povu', variance='var'))
    for name in aggregated:
        mse = np.mean(aggregated[name][0])
        pove = np.mean(aggregated[name][1])
        povu = np.mean(aggregated[name][2])
        variance = np.mean(aggregated[name][3])
        print('{name:8}  {mse:>10.6}  {pove:>10.6}  {povu:>10.6}  {variance:>10.6}'.format(
            name=name, mse=mse, pove=pove, povu=povu, variance=variance))
    print('')
    print('')


def sentence_predictions(paths, variation_set_name, training_variation, aux_loss, num_runs):
    output_dir = os.path.join(paths.base_path, 'bert', variation_set_name, task_hash(set(training_variation)))
    result = dict()
    for index_run in range(num_runs):
        validation_json_path = os.path.join(output_dir, 'run_{}'.format(index_run), 'output_validation.json')
        with open(validation_json_path, 'rt') as validation_json_file:
            output_results = json.load(validation_json_file)
        for output_result in output_results:
            name = output_result['name']
            if name not in training_variation and name not in aux_loss:
                continue
            data_key, unique_id = output_result['data_key'], output_result['unique_id']
            if data_key not in result:
                result[data_key] = dict()
            if unique_id not in result[data_key]:
                result[data_key][unique_id] = dict()
            if name not in result[data_key][unique_id]:
                result[data_key][unique_id][name] = list()
            result[data_key][unique_id][name].append(output_result)

    return result
