import re

from .text_wrap_manager import TextWrapManager
from .text_grid_style import TextGridStyle


__all__ = ['TextCell', 'TextCellRenderer']


class TextCell:
    """
    A single cell within a table.
    """

    def __init__(self, text, cell_style=None):
        """
        Create an instance of a cell
        :param text: The text to put into the cell. This can be a string, a TextWrapManager instance or a list of
        strings/TextWrapManager instances. For details of the behavior for each, see comments on set_text.
        :param cell_style: The style for this cell.
        """
        self.style = cell_style or TextGridStyle()
        self._lines = None
        if text is None:
            text = ''
        self.set_text(text)

    @staticmethod
    def padding_at_width(desired_padding, width, no_break_width):
        """
        Padding is removed when the cell width is too small for lines not to be broken. This function returns the actual
        amount of padding given the desired amount of padding, the width of the cell, and width of the cell at which no
        lines need to be broken apart.
        :param desired_padding: The amount of padding desired.
        :param width: The width of the cell.
        :param no_break_width: The width at which no lines are broken.
        :return: The actual amount of padding given these parameters.
        """
        return max(0, min((width - no_break_width) / 2, desired_padding))

    def desired_column_padding(self):
        """
        The amount of column padding in the style for this cell.
        """
        return max(self.style.get_column_padding(), 0)

    def actual_row_padding(self):
        """
        The amount of row padding for this cell. The desired amount is always the actual amount.
        """
        return max(self.style.get_row_padding(), 0)

    def column_padding(self, width):
        """
        The actual amount of column padding for this cell at the given width.
        :param width: The width for which to calculate the column padding.
        """
        return TextCell.padding_at_width(self.desired_column_padding(), width, self.no_word_break_width())

    def hard_break_lines(self):
        """
        generator which iterates over the hard break lines in this cell (i.e. the lines where the user wants line
        breaks)
        """
        if self._lines is not None:
            for wrap_manager in self._lines:
                yield wrap_manager.text

    def line_count_at_width(self, width):
        """
        The number of lines in this cell at the given width.
        :param width: The width at which to calculate the number of lines.
        """
        if self._lines is None:
            return 0
        result = 0
        for line in self._lines:
            result += line.line_count_at_width(width - self.column_padding(width) * 2)
        return result

    def broken_word_count_at_width(self, width):
        """
        The number of words that need to be broken across lines at the given width.
        :param width: The width at which to calculate the number of broken words.
        """
        if self._lines is None:
            return 0
        result = 0
        for line in self._lines:
            result += line.broken_words_at_width(width - self.column_padding(width) * 2)
        return result

    def no_word_break_width(self):
        """
        The minimum width at which no words are broken across lines.
        """
        if self._lines is None:
            return 0
        result = 0
        for line in self._lines:
            result = max(result, line.no_word_break_width())
        return result

    def no_break_width(self):
        """
        The minimum width at which no line breaks occur.
        """
        if self._lines is None:
            return 0
        result = 0
        for line in self._lines:
            if line.text is not None:
                result = max(result, len(line.text))

        # this includes padding because we will break lines rather than removing padding
        return result + self.desired_column_padding() * 2

    def preferred_width(self):
        """
        The width desired for this cell.
        If an explicit width is set by the user, this is the bigger of the explicit width and the width at which no
        words need to be broken across lines. Otherwise, this is the width at which no lines need to be broken.
        """
        if self.style.get_width() > 0:
            return max(self.no_word_break_width(), self.style.get_width())
        else:
            return self.no_break_width()

    def set_text(self, text):
        """
        Sets the text for this cell. Text can be a string, an instance of a TextWrapManager, or an iterable of
        strings and TextWrapManager instances. Every string will be split at '\n'. And using a string with '\n' in it
        is equivalent to using an iterable of the strings resulting from the split. Each item in the list of strings
        and TextWrapManagers is considered a hard break - i.e. these line breaks will be kept when the cell is
        rendered.
        :param text: The text to set.
        """
        self._lines = list()
        if isinstance(text, str) or isinstance(text, TextWrapManager):
            line_iterable = [text]
        else:
            line_iterable = text
        split_lines = list()
        for line in line_iterable:
            if isinstance(line, str):
                split_lines.extend(line.split('\n'))
            else:
                split_lines.append(line)
        for line in split_lines:
            if not isinstance(line, TextWrapManager):
                line = TextWrapManager(line, self.style.get_line_indent(), self.style.get_line_style())
            self._lines.append(line)

    def single_clean_string(self):
        """
        Gets the text of this cell as a single string with a single space separating lines and with all whitespace
        between non-whitespace replaced by a single space
        """
        result = ''
        for line in self.hard_break_lines():
            if line is not None and len(line) > 0:
                if len(result) > 0:
                    result += ' '
                result += line

        return re.sub(r'\s', ' ', result)

    def _rendered_lines(self, width):
        result = list()
        if self._lines is None or len(self._lines) == 0:
            return result

        for line in self._lines:
            result.extend(line.lines(width))

        return result

    def make_renderer(self, width):
        padding = ' ' * self.column_padding(width)
        lines = [padding + line + padding for line in self._rendered_lines(width - self.column_padding(width) * 2)]
        return TextCellRenderer(self, lines, width)


class TextCellRenderer:

    def __init__(self, cell, cell_lines, width):
        self._width = width
        self._lines = cell_lines
        self._row_padding = cell.actual_row_padding()
        self._style = cell.style

    def __len__(self):
        return self._row_padding * 2 + len(self._lines)

    def line_at(self, index):
        if index < self._row_padding:
            return ' ' * self._width
        index -= self._row_padding
        if index < len(self._lines):
            return self._lines[index]
        return ' ' * self._width

    def render(self, writer, style_handler, index):
        style_handler.push_style(self._style)
        writer.write(self.line_at(index))
        style_handler.pop_style()
