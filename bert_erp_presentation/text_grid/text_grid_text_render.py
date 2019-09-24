import sys
import shutil
import dataclasses
from typing import Sequence
import math

from .text_grid_style import TextGridStyle, EmptyStyleHandler


__all__ = ['border_characters', 'render_text_grid_as_text', 'write_text_grid_to_console']


@dataclasses.dataclass(frozen=True)
class _BorderChars:
    down_and_right: str = chr(0x250c)
    horizontal: str = chr(0x2500)
    down_and_left: str = chr(0x2510)
    vertical: str = chr(0x2502)
    up_and_right: str = chr(0x2514)
    up_and_left: str = chr(0x2518)
    vertical_and_horizontal: str = chr(0x253c)
    down_and_horizontal: str = chr(0x252c)
    up_and_horizontal: str = chr(0x2534)
    vertical_and_right: str = chr(0x251c)
    vertical_and_left: str = chr(0x2524)

    @staticmethod
    def for_encoding(encoding):
        border_characters_ = _BorderChars()
        needs_fallback = False
        for border_character in dataclasses.astuple(border_characters_):
            try:
                border_character.encode(encoding)
            except UnicodeEncodeError:
                needs_fallback = True
                break
        if needs_fallback:
            return _BorderChars(
                down_and_right='+',
                horizontal='-',
                down_and_left='+',
                vertical='|',
                up_and_right='+',
                up_and_left='+',
                vertical_and_horizontal='+',
                down_and_horizontal='+',
                up_and_horizontal='+',
                vertical_and_right='+',
                vertical_and_left='+')
        return border_characters_


border_characters = _BorderChars.for_encoding(sys.stdout.encoding)


@dataclasses.dataclass
class _WidthInfo:
    index_column: int
    width: int
    line_count: int = 0
    multi_line_cells: int = 0
    broken_word_count: int = 0
    broken_column_count: int = 0
    non_user_count: int = 0
    under_padding: int = 0

    @staticmethod
    def from_cell(text_cell, index_column, width):
        explicit_width = text_cell.style.get_width()
        line_count = text_cell.line_count_at_width(width)
        broken_word_count = text_cell.broken_word_count_at_width(width)
        return _WidthInfo(
            index_column=index_column,
            width=width,
            line_count=line_count,
            multi_line_cells=line_count > 1,
            broken_word_count=broken_word_count,
            broken_column_count=broken_word_count > 1,
            non_user_count=1 if explicit_width is not None and explicit_width > 0 and explicit_width != width else 0,
            under_padding=text_cell.colum_padding(width) < text_cell.desired_column_padding())

    @staticmethod
    def combine(items: Sequence['_WidthInfo']) -> '_WidthInfo':
        result = None
        for i, x in enumerate(items):
            if i == 0:
                result = _WidthInfo(x.index_column, x.width)
            if x.index_column != result.index_column or x.width != result.width:
                raise ValueError('Invalid combination')
            result.line_count += x.line_count
            result.multi_line_cells += x.multi_line_cells
            result.broken_word_count += x.broken_word_count
            result.broken_column_count = max(result.broken_column_count, x.broken_column_count)
            result.non_user_count += x.non_user_count
            result.under_padding += x.under_padding
        return result

    @staticmethod
    def diff(x, y):
        if x.index_column != y.index_column:
            raise ValueError('Invalid diff')
        return _WidthInfo(
            x.index_column,
            x.width,
            x.line_count - y.line_count,
            x.multi_line_cells - y.multi_line_cells,
            x.broken_word_count - y.broken_word_count,
            x.broken_column_count - y.broken_column_count,
            x.non_user_count - y.non_user_count,
            x.under_padding - y.under_padding)


class _TextGridTextRender:

    def __init__(self, text_grid, width=None, table_style=None, style_handler=None):
        self.text_grid, self.width, self.table_style, self.style_handler = text_grid, width, table_style, style_handler
        if self.table_style is None:
            self.table_style = TextGridStyle()

        if self.style_handler is None:
            self.style_handler = EmptyStyleHandler()

        if self.width is None:
            self.width = type(self).no_break_width(self.text_grid, self.table_style)
        minimum_width = type(self).minimum_width(self.text_grid, self.table_style)
        if self.width < minimum_width:
            raise ValueError('minimum width of table is {0}'.format(minimum_width))

        explicit_width = self.table_style.get_width()
        if explicit_width > 0:
            self.width = min(explicit_width, width)

        self.column_widths = self._compute_column_widths()

    @staticmethod
    def no_break_width(text_grid, table_style):
        """
        The minimum width at which no lines in this table are wrapped.
        """

        if table_style is None:
            table_style = TextGridStyle()

        column_widths = [0] * text_grid.column_count
        for spec in text_grid.iterate_specs():
            if spec.num_columns > 1:
                continue
            column_widths[spec.index_grid_left] = max(
                column_widths[spec.index_grid_left], spec.value.no_break_width())

        for spec in text_grid.iterate_specs():
            if spec.num_columns == 1:
                continue
            current_span_width = 0
            non_zero_column = -1
            for k in range(spec.num_columns):
                current_span_width += column_widths[spec.index_grid_left + k]
                if column_widths[spec.index_grid_left + k] > 0:
                    non_zero_column = k

                # add additional width to a non-zero width column if possible
                # and to the first column otherwise to keep border calculation correct
                if current_span_width < spec.value.no_break_width():
                    if non_zero_column < 0:
                        column_widths[spec.index_grid_left] += spec.value.no_break_width() - current_span_width
                    else:
                        column_widths[spec.index_grid_left + non_zero_column] += \
                            spec.value.no_break_width() - current_span_width

        result = 0
        for i in range(0, len(column_widths)):
            result += column_widths[i]
            if table_style.get_border() is not None and i > 0 and column_widths[i - 1] > 0:
                result += 1

        # outside borders
        if table_style.get_border() is not None:
            result += 2

        left_margin = max(table_style.get_left_margin(), 0)
        right_margin = max(table_style.get_right_margin(), 0)

        return result + left_margin + right_margin

    @staticmethod
    def minimum_width(text_grid, table_style):
        """
        The minimum width this table can possibly take if the widths of all columns are minimized.
        """

        if table_style is None:
            table_style = TextGridStyle()

        left_margin = max(table_style.get_left_margin(), 0)
        right_margin = max(table_style.get_right_margin(), 0)
        if table_style.get_border() is not None:
            # 1 * column_count = space to write data
            # (1 * column_count) + 1 = space to write borders
            return text_grid.column_count * 2 + 1 + left_margin + right_margin
        else:
            return text_grid.column_count + left_margin + right_margin

    def _compute_column_widths(self):

        left_margin = max(self.table_style.get_left_margin(), 0)
        right_margin = max(self.table_style.get_right_margin(), 0)
        usable_width = self.width - left_margin - right_margin
        if self.table_style.get_border() is not None and self.text_grid.column_count > 0:
            usable_width -= (self.text_grid.column_count - 1) + 2

        column_widths = [0] * self.text_grid.column_count

        for spec in self.text_grid.iterate_specs():
            if spec.num_columns > 1:
                continue
            column_widths[spec.index_grid_left] = max(column_widths[spec.index_grid_left], spec.value.preferred_width())

        self._adjust_widths_for_multi_column_cells(column_widths)

        total_preferred_width = sum(column_widths)
        if usable_width < total_preferred_width:
            self._reduce_widths(column_widths, usable_width)
        else:
            self._distribute_widths(column_widths, usable_width)

        return column_widths

    def _adjust_widths_for_multi_column_cells(self, column_widths):
        for spec in self.text_grid.iterate_specs():
            if spec.num_columns == 1:
                continue
            current_span_preferred = 0
            for k in range(spec.num_columns):
                current_span_preferred += column_widths[spec.index_grid_left + k]

            if current_span_preferred < spec.value.preferred_width():
                new_span_preferred = 0
                diff = spec.value.preferred_width() - current_span_preferred
                for k in range(spec.num_columns):
                    if current_span_preferred == 0:
                        proportion = 0
                    else:
                        proportion = column_widths[spec.index_grid_left + k] / current_span_preferred
                    column_widths[spec.index_grid_left + k] += int(math.floor(proportion * diff))
                    new_span_preferred += column_widths[spec.index_grid_left + k]

                if new_span_preferred < spec.value.preferred_width():
                    column_widths[spec.index_grid_left + spec.num_columns - 1] += \
                        spec.value.preferred_width() - new_span_preferred

    def _distribute_widths(self, column_widths, usable_width):

        non_explicit_total = 0
        explicit_total = 0

        explicit_width_columns = self._get_explicit_width_columns()
        for i in range(0, len(column_widths)):
            if explicit_width_columns[i]:
                explicit_total += column_widths[i]
            else:
                non_explicit_total += column_widths[i]

        if non_explicit_total == 0:
            for i in range(0, len(column_widths)):
                explicit_width_columns[i] = False
            non_explicit_total = explicit_total
            explicit_total = 0

        to_allocate = usable_width - explicit_total - non_explicit_total
        new_total = explicit_total
        for i in range(0, len(column_widths)):
            if explicit_width_columns[i]:
                continue
            proportion = column_widths[i] / non_explicit_total
            column_widths[i] = int(math.floor(column_widths[i] + proportion * to_allocate))
            new_total += column_widths[i]

        if new_total < usable_width and len(column_widths) > 0:
            column_widths[-1] += usable_width - new_total

    def _get_explicit_width_columns(self):
        result = [False] * self.text_grid.column_count
        for spec in self.text_grid.iterate_specs():
            cell_width = spec.value.style.get_width()
            if cell_width is not None and cell_width > 0:
                for k in range(spec.index_grid_left, spec.index_grid_left + spec.num_columns):
                    result[k] = True
        return result

    def _reduce_widths(self, column_widths, usable_width):
        current_width = sum(column_widths)
        while current_width > usable_width:
            index_column_to_reduce = self._next_reduction_column_index(column_widths)
            current_width -= 1
            column_widths[index_column_to_reduce] -= 1

    def _next_reduction_column_index(self, column_widths):
        column_info_current = dict()
        column_info_target = dict()
        for spec in self.text_grid.iterate_specs():
            for k in range(spec.index_grid_left, spec.num_columns + spec.index_grid_left):
                if column_widths[k] == 1:
                    continue

                column_info_current[k] = _WidthInfo.from_cell(spec.value, k, column_widths[k])
                column_info_target[k] = _WidthInfo.from_cell(spec.value, k, column_widths[k] - 1)

        for i in range(len(column_widths)):
            if i not in column_info_current:
                return i

        # noinspection PyTypeChecker
        candidates = [_WidthInfo.diff(
            _WidthInfo.combine(column_info_target[i]), _WidthInfo.combine(column_info_current[i]))
            for i in column_info_target]

        candidates = sorted(candidates, key=lambda w: (
            w.non_user_count, w.broken_word_count, w.broken_column_count, w.under_padding, w.line_count,
            w.multi_line_cells, -w.width))

        return candidates[0].index_column

    def render(self, writer):

        renderers = dict()
        last_row_to_key = dict()
        positions = dict()

        for spec in self.text_grid.iterate_specs():
            key = spec.index_grid_top, spec.index_grid_left
            renderers[key] = spec.value.make_renderer(self.column_widths[spec.index_grid_left])
            positions[key] = 0
            last_row_to_key[(spec.index_grid_top + spec.num_rows - 1, spec.index_grid_left)] = key

        for index_row in range(self.text_grid.row_count):
            contains_start_row = False
            for index_column in range(self.text_grid.column_count):
                key = index_row, index_column
                if key in renderers:
                    contains_start_row = True
                    break
            if contains_start_row and self.table_style.get_border() is not None:
                self._write_row_border(writer, index_row, renderers, positions)

            while True:
                is_more_to_write = False
                for index_column in range(self.text_grid.column_count):
                    last_row_key = index_row, index_column
                    if last_row_key in last_row_to_key:
                        key = last_row_to_key[last_row_key]
                        if positions[key] < len(renderers[key]):
                            is_more_to_write = True
                        break
                if not is_more_to_write:
                    break

                left_margin = max(self.table_style.get_left_margin(), 0)
                writer.write(' ' * left_margin)

                if self.table_style.get_border() is not None:
                    writer.write(border_characters.vertical)

                has_written_column = False
                for index_column in range(self.text_grid.column_count):
                    key = index_row, index_column
                    if key in renderers:
                        renderer = renderers[key]
                        if has_written_column:
                            if self.table_style.get_border() is not None:
                                writer.write(border_characters.vertical)
                        renderer.render(writer, self.style_handler, positions[key])
                        positions[key] += 1
                        has_written_column = True

                if self.table_style.get_border() is not None:
                    writer.write(border_characters.vertical)
                writer.write('\n')

        if self.table_style.get_border() is not None:
            self._write_row_border(writer, self.text_grid.row_count, renderers, positions)

    def _write_row_border(self, writer, index_row, renderers, positions):
        left_margin = max(self.text_grid.table_style.get_left_margin(), 0)
        writer.write(' ' * left_margin)
        index_cells = min(index_row, self.text_grid.num_rows - 1)
        for i in range(self.text_grid.num_columns):
            spec = self.text_grid.get_spec(index_cells, i)
            renderer = None
            if spec is not None:
                renderer = renderers[(spec.index_grid_top, spec.index_grid_left)]
            if index_row != self.text_grid.num_rows and spec is not None and spec.index_grid_top < index_row:
                is_border, border_char = self._get_border_char(index_row, i)
                if is_border:
                    writer.write(border_char)
                if renderer is not None:
                    renderer.render(writer, self.style_handler, positions[(index_cells, i)])
                    positions[(index_cells, i)] += 1
            else:
                num_columns = spec.num_columns if spec is not None else 1
                for j in range(num_columns):
                    is_border, border_char = self._get_border_char(index_row, i + j)
                    if is_border:
                        writer.write(border_char)
                    writer.write(border_characters.horizontal * self.column_widths[i + j])
        is_end_border, end_border_char = self._get_border_char(index_row, self.text_grid.column_count)
        if is_end_border:
            writer.write(end_border_char)
        writer.write('\n')

    def _get_border_char(self, index_row, index_column):
        is_left = index_column > 0 \
            and not self.text_grid.is_part_of_previous_row(index_row, index_column - 1)
        is_right = index_column < self.text_grid.column_count \
            and not self.text_grid.is_part_of_previous_row(index_row, index_column)
        is_up = index_row > 0 \
            and not self.text_grid.is_part_of_previous_column(index_row - 1, index_column)
        is_down = index_row < self.text_grid.row_count \
            and not self.text_grid.is_part_of_previous_column(index_row, index_column)

        if is_left and is_right:
            if is_up and is_down:
                return True, border_characters.vertical_and_horizontal
            if is_up:
                return True, border_characters.up_and_horizontal
            if is_down:
                return True, border_characters.down_and_horizontal
            return True, border_characters.horizontal

        if is_left:
            if is_up and is_down:
                return True, border_characters.vertical_and_left
            if is_up:
                return True, border_characters.up_and_left
            if is_down:
                return True, border_characters.down_and_left
            return False, '\0'

        if is_right:
            if is_up and is_down:
                return True, border_characters.vertical_and_right
            if is_up:
                return True, border_characters.up_and_right
            if is_down:
                return True, border_characters.down_and_right
            return False, '\0'

        if not is_up and not is_down:
            return False, '\0'
        return True, border_characters.vertical


def render_text_grid_as_text(writer, text_grid, width=None, table_style=None, style_handler=None):
    if width == 'console' or width == 'tight':
        console_size = shutil.get_terminal_size((80, 20)).columns
        if width == 'tight':
            width = min(console_size - 1, _TextGridTextRender.no_break_width(text_grid, table_style))
        else:
            width = console_size - 1
    renderer = _TextGridTextRender(text_grid, width, table_style, style_handler)
    renderer.render(writer)


def write_text_grid_to_console(text_grid, width=None, table_style=None, style_handler=None):
    render_text_grid_as_text(sys.stdout, text_grid, width, table_style, style_handler)
