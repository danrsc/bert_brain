from matplotlib.legend_handler import HandlerNpoints
from matplotlib.lines import Line2D
import numpy as np


class DisplayName:

    def __init__(self, name_dict=None):
        self._name_dict = name_dict

    def add(self, original_names, new_name):
        self._name_dict[original_names] = new_name

    def replace(self, current):
        current_indices = dict()
        for idx, n in enumerate(current):
            current_indices[n] = idx
        for check in self._name_dict:
            if all(item in current_indices for item in check):
                min_index = min(current_indices[item] for item in check)
                for item in check:
                    del current_indices[item]
                current_indices[self._name_dict[check]] = min_index
        return tuple(sorted(current_indices, key=lambda k: current_indices[k]))


def make_label(display_name, self_key, label_key):

    if isinstance(label_key, str):
        label_key = (label_key,)

    display_key = self_key
    if self_key.endswith(' (c)') or self_key.endswith(' (s)'):
        display_key = display_key[:-len(' (c)')]

    label_key = display_name.replace(label_key)
    non_self_label_part = [sub_k.upper() for sub_k in label_key if sub_k != display_key]
    if len(non_self_label_part) > 0:
        non_self_label_part = '(' + ', '.join(non_self_label_part) + ')' \
            if len(non_self_label_part) > 1 else non_self_label_part[0]
        return display_key.upper() + ' + ' + non_self_label_part
    else:
        return display_key.upper()


def significance_metro(ax, source_tick, destination_ticks, offset_per_tick, offset_index=None, label=None):
    tick_limits = ax.get_xlim()
    offset_per_tick = offset_per_tick * (tick_limits[1] - tick_limits[0])

    min_dest = min(destination_ticks)
    max_dest = max(destination_ticks)
    if source_tick > max_dest:
        max_dest = min_dest
    x = offset_per_tick * source_tick if offset_index is None else offset_per_tick * offset_index
    legend_handle = ax.plot([x, x], [source_tick, max_dest], c='black', label=label)[0]
    ax.plot([x], [source_tick], marker='o', markerfacecolor='black', markeredgecolor='black')
    for destination_tick in destination_ticks:
        ax.plot([x], [destination_tick], marker='o', markerfacecolor='white', markeredgecolor='black')
    return legend_handle


class SignificanceMetroLegendHandler(HandlerNpoints):
    """
    Handler for `.Line2D` instances.
    """
    def __init__(self, marker_pad=0.3, **kw):
        """
        Parameters
        ----------
        marker_pad : float
            Padding between points in legend entry.

        numpoints : int
            Number of points to show in legend entry.

        Notes
        -----
        Any other keyword arguments are given to `HandlerNpoints`.
        """
        HandlerNpoints.__init__(self, marker_pad=marker_pad, numpoints=2, **kw)

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):

        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)

        ydata = np.full_like(xdata, ((height - ydescent) / 2))
        legline = Line2D(xdata, ydata)

        self.update_prop(legline, orig_handle, legend)
        legline.set_drawstyle('default')
        legline.set_marker("")

        source_marker = Line2D(
            [xdata_marker[0]], [ydata[0]], marker='o', markerfacecolor='black', markeredgecolor='black')
        dest_marker = Line2D(
            [xdata_marker[1]], [ydata[1]], marker='o', markerfacecolor='white', markeredgecolor='black')
        for marker, facecolor in [(source_marker, 'black'), (dest_marker, 'white')]:
            self.update_prop(marker, orig_handle, legend)
            marker.set_linestyle('None')
            marker.set_marker('o')
            marker.set_markerfacecolor(facecolor)
            marker.set_markeredgecolor('black')
            if legend.markerscale != 1:
                newsz = marker.get_markersize() * legend.markerscale
                marker.set_markersize(newsz)
        # we don't want to add this to the return list because
        # the texts and handles are assumed to be in one-to-one
        # correspondence.
        legline._source_marker = source_marker
        legline._dest_marker = dest_marker

        legline.set_transform(trans)
        source_marker.set_transform(trans)
        dest_marker.set_transform(trans)

        return [legline, source_marker, dest_marker]


def get_significance_metro_lines(bar_group, tick_offset=0, is_top_to_bottom=False, ticks=None):

    sig_lines = list()
    if ticks is None:
        ticks = np.arange(len(bar_group))

    if bar_group.significance_pairs is not None:
        if is_top_to_bottom:
            for i in range(1, len(bar_group)):
                if bar_group.hidden is not None and i in bar_group.hidden:
                    continue
                j_points = list()
                for j in range(i):
                    if bar_group.hidden is not None and j in bar_group.hidden:
                        continue
                    if (i, j) in bar_group.significance_pairs or (j, i) in bar_group.significance_pairs:
                        j_points.append(ticks[j] + tick_offset)
                if len(j_points) > 0:
                    sig_lines.append((ticks[i] + tick_offset, j_points, len(bar_group) - i))
        else:
            for i in range(len(bar_group) - 1):
                if bar_group.hidden is not None and i in bar_group.hidden:
                    continue
                j_points = list()
                for j in range(i + 1, len(bar_group)):
                    if bar_group.hidden is not None and j in bar_group.hidden:
                        continue
                    if (i, j) in bar_group.significance_pairs or (j, i) in bar_group.significance_pairs:
                        j_points.append(ticks[j] + tick_offset)
                if len(j_points) > 0:
                    sig_lines.append((ticks[i] + tick_offset, j_points, 1 + i))
    return sig_lines


def keyed_colors(keys, modulo=False):
    from matplotlib import pyplot as plt
    prop_cycle = plt.rcParams['axes.prop_cycle']
    temp_colors = prop_cycle.by_key()['color']
    if len(keys) > len(temp_colors) and not modulo:
        raise ValueError('Too many keys specified')
    return dict((k, temp_colors[i % len(temp_colors)]) for i, k in enumerate(keys))


class BarGroup:

    def __init__(
            self,
            labels,
            values,
            index_legend_bar=None,
            color=None,
            legend_label=None,
            significance_pairs=None,
            hidden=None,
            hidden_value=0,
            train_keys=None):
        self.labels = labels
        self.values = values
        self.train_keys = train_keys
        self.color = color
        self.legend_label = legend_label
        self.significance_pairs = significance_pairs
        self.index_legend_bar = index_legend_bar
        self.hidden = hidden
        self.hidden_value = hidden_value

    def sort(self, key, call_with_train_keys=False):
        items = zip(self.labels, self.values, self.train_keys) \
            if call_with_train_keys else zip(self.labels, self.values)
        sort_order = [i for i, _ in sorted(enumerate(items), key=lambda i_kv: key(i_kv[1]))]
        self.labels = [self.labels[i] for i in sort_order]
        self.values = [self.values[i] for i in sort_order]
        if self.train_keys is not None:
            self.train_keys = [self.train_keys[i] for i in sort_order]
        if isinstance(self.color, (list, tuple)):
            self.color = [self.color[i] for i in sort_order]
        invert_sort = [i for i, _ in sorted(enumerate(sort_order), key=lambda item: item[1])]
        if self.significance_pairs is not None:
            self.significance_pairs = set((invert_sort[i], invert_sort[j]) for i, j in self.significance_pairs)
        if self.index_legend_bar is not None:
            self.index_legend_bar = invert_sort[self.index_legend_bar]
        if self.hidden is not None:
            self.hidden = set(invert_sort[i] for i in self.hidden)

    def find(self, train_keys):
        if self.train_keys is None:
            raise ValueError('train_keys not set, filtering not available')
        have_as_sets = [set(k) for k in self.train_keys]

        def _find(key):
            k = set(key)
            for i, h in enumerate(have_as_sets):
                if h == k:
                    return i
            raise ValueError('Unable to find key: {}'.format(key))

        return [_find(key_) for key_ in train_keys]

    def filter(self, train_keys):
        indices_keep = self.find(train_keys)
        inverted = [None] * len(self)
        for new_index, old_index in enumerate(indices_keep):
            inverted[old_index] = new_index
        self.labels = [self.labels[i] for i in indices_keep]
        self.values = [self.values[i] for i in indices_keep]
        self.train_keys = [self.train_keys[i] for i in indices_keep]
        if isinstance(self.color, (list, tuple)):
            self.color = [self.color[i] for i in indices_keep]

        significance_pairs = set()
        if self.significance_pairs is not None:
            for i, j in self.significance_pairs:
                if inverted[i] is not None and inverted[j] is not None:
                    significance_pairs.add((inverted[i], inverted[j]))
            self.significance_pairs = significance_pairs
        if self.index_legend_bar is not None:
            self.index_legend_bar = inverted[self.index_legend_bar]
        if self.hidden is not None:
            hidden = set()
            for i in self.hidden:
                if inverted[i] is not None:
                    hidden.add(i)
            self.hidden = hidden

    def __len__(self):
        return len(self.labels)


def make_bar_group(metrics, p_values, bhy_thresh, key_to_color, make_label_fn, key_filter=None):
    train_keys = [k for k in metrics]
    index_self = [i for i, k in enumerate(train_keys) if len(k) == 1][0]
    self_key = train_keys[index_self][0]

    if key_filter is not None:
        train_keys = [k for k in train_keys if key_filter(k)]

    means = np.array([np.mean(metrics[k]) for k in train_keys])

    if key_to_color is not None:
        color = key_to_color[self_key]
        colors = [color] * len(means)
    else:
        colors = None
    labels = [make_label_fn(self_key, k) for k in train_keys]

    significance_pairs = set()
    for i in range(1, len(train_keys)):
        for j in range(i):
            if p_values[(train_keys[i], train_keys[j])] <= bhy_thresh:
                significance_pairs.add((i, j))

    return BarGroup(
        labels,
        means,
        train_keys=train_keys,
        index_legend_bar=index_self,
        color=colors,
        legend_label=make_label_fn(self_key, self_key),
        significance_pairs=significance_pairs)


def bar_eval_key(ax, metrics, p_values, bhy_thresh, make_label_fn):

    bar_group = make_bar_group(metrics, p_values, bhy_thresh, None, make_label_fn)
    bar_group.sort(lambda kv: (len(kv[2]), kv[1]), call_with_train_keys=True)

    # err = np.array([np.std(metrics[k]) for k in train_keys])
    values = bar_group.values
    labels = bar_group.labels
    if bar_group.hidden is not None:
        values = [v if i not in bar_group.hidden else bar_group.hidden_value for i, v in enumerate(values)]
        labels = [l if i not in bar_group.hidden else '' for i, l in enumerate(labels)]

    ticks = list()
    last_len = None
    for train_key in bar_group.train_keys:
        is_new_len = len(train_key) > last_len if last_len is not None else False
        last_len = len(train_key)
        if is_new_len:
            ticks.append(ticks[-1] + 2)
        else:
            ticks.append(ticks[-1] + 1 if len(ticks) > 0 else 0)

    bar_handle = ax.barh(ticks, values, tick_label=labels)
    for start, stops, offset_index in get_significance_metro_lines(bar_group, ticks=ticks):
        significance_metro(ax, start, stops, 0.025, offset_index)

    ax.set_title(bar_group.legend_label)
    return bar_handle


def bar_grouped(ax, groups, is_top_to_bottom=False):

    ticks = list()
    tick_labels = list()
    values = list()
    colors = list()

    first_bars = list()
    legend_labels = list()

    sig_lines = list()

    for bar_group in groups:

        group_values = bar_group.values
        group_labels = bar_group.labels
        if bar_group.hidden is not None:
            group_values = [
                v if i not in bar_group.hidden else bar_group.hidden_value for i, v in enumerate(group_values)]
            group_labels = [l if i not in bar_group.hidden else '' for i, l in enumerate(group_labels)]

        legend_labels.append(bar_group.legend_label)
        tick_labels.extend(group_labels)
        values.extend(group_values)
        if bar_group.color is None:
            colors.extend([None] * len(bar_group))
        elif isinstance(bar_group.color, (list, tuple)):
            colors.extend(bar_group.color)
        else:
            colors.extend([bar_group.color] * len(bar_group))

        index_label_bar = bar_group.index_legend_bar
        if index_label_bar is None:
            index_label_bar = 0

        first_bars.append(len(ticks) + index_label_bar)
        ticks.append(0 if len(ticks) == 0 else ticks[-1] + 2)
        tick_offset = ticks[-1]
        for _ in range(len(bar_group) - 1):
            ticks.append(ticks[-1] + 1)

        sig_lines.extend(
            get_significance_metro_lines(bar_group, tick_offset=tick_offset, is_top_to_bottom=is_top_to_bottom))

    if all(c is None for c in colors):
        colors = None
    bar_handle = ax.barh(ticks, values, tick_label=tick_labels, color=colors)
    sig_handle = None
    for start, stops, offset_index in sig_lines:
        sig_handle = significance_metro(ax, start, stops, 0.025, offset_index)

    legend_bars = [bar_handle[i] for i in first_bars]
    if is_top_to_bottom:
        legend_bars = list(reversed(legend_bars))
        legend_labels = list(reversed(legend_labels))
    handler_map = None
    # to force the significance legend marker
    # if sig_handle is None:
    #     sig_handle = ax.plot([0, 0], [0, 0], c='black')[0]
    if sig_handle is not None:
        legend_bars.append(sig_handle)
        legend_labels.append('Sig. diff.')
        handler_map = {sig_handle: SignificanceMetroLegendHandler()}
    ax.legend(legend_bars, legend_labels, handler_map=handler_map)
    return bar_handle


def bar_group_best_for_eval(
        metrics, p_values, bhy_thresh, key_to_color, make_label_fn, best_only=False):
    from matplotlib import colors as mcolors
    index_self = [i for i, k in enumerate(metrics) if len(k) == 1][0]
    train_keys = [k for k in metrics]
    means = np.array([np.mean(metrics[k]) for k in train_keys])
    self_mean = means[index_self]
    means[index_self] = np.nan
    index_best = np.nanargmax(means)
    means[index_self] = self_mean

    result_indices = list()

    best_key = train_keys[index_best]
    shortest = len(best_key)
    for index_alt, alt_key in enumerate(train_keys):
        if index_alt == index_best or index_alt == index_self:
            result_indices.append(index_alt)
        elif not best_only:
            if len(alt_key) <= len(best_key) \
                    and p_values[(alt_key, best_key)] > bhy_thresh >= p_values[(alt_key, train_keys[index_self])]:
                result_indices.append(index_alt)
                shortest = min(shortest, len(alt_key))

    result_indices = [
        i for i in result_indices if i == index_best or i == index_self or len(train_keys[i]) == shortest]

    color = tuple(mcolors.to_rgba(key_to_color[train_keys[index_self][0]]))

    colors = [color] * len(result_indices)
    significance_pairs = set()
    for i in range(1, len(result_indices)):
        for j in range(i):
            if p_values[(train_keys[result_indices[i]], train_keys[result_indices[j]])] <= bhy_thresh:
                significance_pairs.add((i, j))

    index_self_in_group = [i for i, idx in enumerate(result_indices) if idx == index_self][0]

    return BarGroup(
        [make_label_fn(train_keys[index_self][0], train_keys[i]) for i in result_indices],
        [means[i] for i in result_indices],
        train_keys=train_keys,
        legend_label=make_label_fn(train_keys[index_self][0], train_keys[index_self]),
        color=colors,
        significance_pairs=significance_pairs,
        index_legend_bar=index_self_in_group)
