import os
from .experiments import singleton_variation, task_hash
from .result_output import read_predictions


__all__ = ['sentence_predictions']


def sentence_predictions(paths, variation_name, key):
    (variation_name, _), settings = singleton_variation(variation_name)
    output_dir = os.path.join(paths.result_path, variation_name, task_hash(settings))
    result = dict()
    has_warned = False
    for index_run in range(settings.num_runs):
        validation_npz_path = os.path.join(output_dir, 'run_{}'.format(index_run), 'output_validation.npz')
        if not os.path.exists(validation_npz_path):
            if not has_warned:
                print('warning: results are incomplete. Some runs not found')
            has_warned = True
            continue
        output_results = read_predictions(validation_npz_path)
        for output_result in output_results[key]:
            if output_result.unique_id not in result:
                result[output_result.unique_id] = list()
            result[output_result.unique_id].append(output_result)
    return result
