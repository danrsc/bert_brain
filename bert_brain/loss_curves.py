import dataclasses
import os
from typing import Tuple

import numpy as np

from .experiments import named_variations, task_hash
from .result_output import read_loss_curve


__all__ = [
    'average_unique_epochs_within_loss_curves',
    'average_unique_steps_within_loss_curves',
    'LossCurve',
    'loss_curves_for_variation']


def average_unique_steps_within_loss_curves(curves):
    for curve in curves:
        unique_steps = np.unique(curve.steps)
        step_values = list()
        step_epochs = list()
        for step in unique_steps:
            step_values.append(np.nanmean(curve.values[curve.steps == step]))
            step_epochs.append(curve.epochs[curve.steps == step][0])
        curve.steps = unique_steps
        curve.epochs = np.array(step_epochs)
        curve.values = np.array(step_values)


def average_unique_epochs_within_loss_curves(curves):
    for curve in curves:
        unique_epochs = np.unique(curve.epochs)
        epoch_values = list()
        for epoch in unique_epochs:
            epoch_values.append(np.nanmean(curve.values[curve.epochs == epoch]))
        curve.steps = unique_epochs
        curve.epochs = unique_epochs
        curve.values = np.array(epoch_values)


def average_over_runs(curves, run_key=None):
    aggregate = dict()
    for curve in curves:
        run = -1 if run_key is None else run_key(curve.index_run)
        if run not in aggregate:
            aggregate[run] = dict()
        if (curve.key, curve.train_eval_kind) not in aggregate[run]:
            aggregate[run][(curve.key, curve.train_eval_kind)] = list()
        aggregate[run][(curve.key, curve.train_eval_kind)].append(curve)
    result = list()
    for run in aggregate:
        for curve_key, train_eval_kind in aggregate[run]:
            steps = dict()
            for curve in aggregate[run][(curve_key, train_eval_kind)]:
                for s, e, v in zip(curve.steps, curve.epochs, curve.values):
                    if (s, e) not in steps:
                        steps[s, e] = list()
                    steps[s, e].append(v)
            curve_steps = list()
            curve_epochs = list()
            curve_values = list()
            for s, e in sorted(steps):
                curve_steps.append(s)
                curve_epochs.append(e)
                curve_values.append(np.mean(steps[s, e]))
            result.append(dataclasses.replace(
                aggregate[run][(curve_key, train_eval_kind)][0],
                index_run=run,
                epochs=np.array(curve_epochs),
                steps=np.array(curve_steps),
                values=np.array(curve_values)))
    return result


@dataclasses.dataclass
class LossCurve:
    training_variation: Tuple[str, ...]
    train_eval_kind: str
    index_run: int
    key: str
    epochs: np.ndarray
    steps: np.ndarray
    values: np.ndarray


def loss_curves_for_variation(paths, variation_set_name):
    named_settings = named_variations(variation_set_name)

    def read_curve(kind, variation_name_, settings_, index_run_):
        file_name = 'train_curve.npz' if kind == 'train' else 'validation_curve.npz'
        output_dir = os.path.join(paths.result_path, variation_name_, task_hash(settings_))
        curve_path = os.path.join(output_dir, 'run_{}'.format(index_run_), file_name)
        result_ = list()
        if os.path.exists(curve_path):
            curve = read_loss_curve(curve_path)
            for key in curve:
                result_.append(
                    LossCurve(
                        settings_.all_loss_tasks, kind, index_run_, key, curve[key][0], curve[key][1], curve[key][2]))
        return result_

    result = list()
    for variation_name, training_variation_name in named_settings:
        settings = named_settings[(variation_name, training_variation_name)]
        for index_run in range(settings.num_runs):
            result.extend(read_curve('train', variation_name, settings, index_run))
            result.extend(read_curve('validation', variation_name, settings, index_run))

    return result
