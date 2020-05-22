import warnings

import numpy as np

from bert_brain import read_variation_results
from ocular import TextGrid, TextWrapStyle, write_text_grid_to_console


output_order_regression = (
    'r',         # correlation
    'mse',       # mean squared error
    'mae',       # mean absolute error
    'pove',      # proportion of variance explained
    'povu',      # proportion of variance unexplained
    'podu',      # proportion of mean absolute deviation unexplained
    'pode',      # proportion of mean absolute deviation explained
    'variance',
    'mad',       # mean absolute deviation
    'r_seq')     # avg (over batch) of sequence correlation values (i.e. correlation within a sequence)

output_order_classifier = (
    'xent',      # cross entropy
    'acc',       # accuracy
    'macc',      # mode accuracy - the accuracy one would get if one picked the mode
    'poma',      # proportion of mode accuracy; < 1 is bad
    'racc',      # rank accuracy
    'prec',      # precision
    'rec',       # recall
    'f1')


def print_variation_results(paths, variation_set_name, index_run=None, field_precision=2, **loss_handler_kwargs):

    aggregated, count_runs, settings = read_variation_results(
        paths, variation_set_name, index_run=index_run, **loss_handler_kwargs)

    for output_order in (output_order_regression, output_order_classifier):
        metrics = list()
        for metric in output_order:
            if any(metric in aggregated[name] for name in aggregated):
                metrics.append(metric)

        text_grid = TextGrid()
        text_grid.append_value('name', column_padding=2)
        for metric in metrics:
            text_grid.append_value(metric, line_style=TextWrapStyle.right_justify, column_padding=2)
        text_grid.next_row()
        value_format = '{' + ':.{}f'.format(field_precision) + '}'
        for name in aggregated:
            if not any(metric in aggregated[name] for metric in metrics):
                continue
            text_grid.append_value(name, column_padding=2)
            for metric in metrics:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    value = np.nanmean(aggregated[name].values(metric)) if metric in aggregated[name] else np.nan
                text_grid.append_value(
                    value_format.format(value), line_style=TextWrapStyle.right_justify, column_padding=2)
            text_grid.next_row()

        training_variation_name = ', '.join(sorted(settings.all_loss_tasks))
        print('Variation ({} of {} runs found): {}'.format(count_runs, settings.num_runs, training_variation_name))
        write_text_grid_to_console(text_grid, width='tight')
        print('')
        print('')


def text_heat_map_html(words, scores, vmin=None, vmax=None, cmap=None, text_color=None):
    from matplotlib import cm, colors
    cmap = cm.ScalarMappable(colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    fmt = '<span style="background-color:{hex}{text_color}">{word}</span>'
    fmt = fmt.format(hex='{hex}', word='{word}', text_color='' if text_color is None else ';color:{text_color}')
    word_colors = cmap.to_rgba(scores)
    return '&nbsp;'.join(
        [fmt.format(word=w, hex=colors.to_hex(c), text_color=text_color) for w, c in zip(words, word_colors)])


def remove_prefix(prefix, x):
    if x.startswith(prefix):
        return x[len(prefix):]
    return x


def remove_hp_fmri_prefix(x):
    return remove_prefix('hp_fmri_', x)


def data_combine_subtract(x, y):
    return x - y


def default_filter_combine(result_query, x, y):
    if result_query.metric == 'pove':
        return np.logical_or(x >= 0.05, y >= 0.05)
    elif result_query.metric == 'k_vs_k':
        return np.logical_or(x >= 0.5, y >= 0.5)
    else:
        return np.full(x.shape, True)


def min_max_default_group_key_fn(result):
    if len(result) == 3:
        return result[0].metric, 'combined'
    return result[0].metric


def min_max_per_group(
        result_queries,
        group_key_fn=min_max_default_group_key_fn,
        data_combine_fn=data_combine_subtract,
        filter_combine_fn=default_filter_combine,
        percentile_min=None,
        percentile_max=None):
    vmin_vmax = dict()
    for result in result_queries:
        key = group_key_fn(result)
        if len(result) == 3:
            result_query, data_1, data_2 = result
            data = data_combine_fn(data_1, data_2)
            if filter_combine_fn is not None:
                data = np.where(filter_combine_fn(result_query, data_1, data_2), data, np.nan)
        else:
            result_query, data = result
        if percentile_min is not None:
            vmin = np.nanpercentile(data, percentile_min, interpolation='higher').item()
        else:
            vmin = np.nanmin(data).item()
        if percentile_max is not None:
            vmax = np.nanpercentile(data, percentile_max, interpolation='lower').item()
        else:
            vmax = np.nanmax(data).item()
        if key not in vmin_vmax:
            vmin_vmax[key] = list(), list()
        vmin_vmax[key][0].append(vmin)
        vmin_vmax[key][1].append(vmax)
    return vmin_vmax
