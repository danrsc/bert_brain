import dataclasses
import math

__author__ = 'drschwar'


__all__ = ['CriticalWidth', 'TextWrapStyle', 'TextWrapManager']


@dataclasses.dataclass(frozen=True)
class CriticalWidth:
    width: int
    line_count: int


class TextWrapStyle:
    """
    Defines how text should be justified and indented.
    """

    def __init__(self):
        return

    indent_first_line = 1
    indent_all_but_first = 2
    indent_all = 3
    indent_first_once_subsequent_twice = 4
    indent_first_twice_subsequent_once = 5
    right_justify = 6
    left_justify = 7
    center = 8

    @staticmethod
    def is_indent_style(style):
        """
        True if the style setting has indentation.
        :param style: The current style setting.
        """
        return (style == TextWrapStyle.indent_first_line or
                style == TextWrapStyle.indent_all_but_first or
                style == TextWrapStyle.indent_all or
                style == TextWrapStyle.indent_first_once_subsequent_twice or
                style == TextWrapStyle.indent_first_twice_subsequent_once)


class TextWrapManager:
    """
    Handles wrapping text at whitespace.
    """

    def __init__(self, text, indent=0, style=TextWrapStyle.left_justify):
        """
        Create a new instance of the TextWrapManager.
        :param text: The text to wrap.
        :param indent: The number of characters to use for each indentation.
        :param style: The line style to use, one of TextWrapManager.Style settings.
        """
        self.text = text
        self._indent = -1
        self._style = None
        self._line_counts_at_width = None
        self._broken_words_at_width = None
        self.set_style(indent, style)
        self.max_word_length = 0
        if self.text is not None and len(self.text) > 0:
            split_text = self.text.split()
            if split_text is not None and len(split_text) > 0:
                self.max_word_length = max([len(x) for x in split_text])

    def _indent_at_pos(self, pos):
        if self._style == TextWrapStyle.indent_all:
            return self._indent
        elif self._style == TextWrapStyle.indent_all_but_first:
            if pos == 0:
                return 0
            return self._indent
        elif self._style == TextWrapStyle.indent_first_line:
            if pos == 0:
                return self._indent
            return 0
        elif self._style == TextWrapStyle.indent_first_once_subsequent_twice:
            if pos == 0:
                return self._indent
            return self._indent * 2
        elif self._style == TextWrapStyle.indent_first_twice_subsequent_once:
            if pos == 0:
                return self._indent * 2
            return self._indent
        return 0

    def _current_line(self, width, start, end):
        if self.text is None:
            return
        index_pos = 0
        result = ''
        style = TextWrapStyle.left_justify if self._style is None else self._style
        if TextWrapStyle.is_indent_style(style):
            current_indent = max(0, self._indent_at_pos(start))
            if width > current_indent:
                result += ' ' * current_indent
        elif style == TextWrapStyle.right_justify:
            num_spaces = (width - (end - start))
            result += ' ' * num_spaces
            index_pos += num_spaces
        elif style == TextWrapStyle.center:
            num_spaces = math.ceil((width - (end - start)) / 2.0)
            result += ' ' * num_spaces
            index_pos += num_spaces
        elif style != TextWrapStyle.left_justify:
            raise ValueError('Unknown value for style: {}'.format(style))

        for index in range(start, end):
            index_pos += 1
            if self.text[index].isspace():
                result += ' '  # transform tabs etc.
            else:
                result += self.text[index]

        result += ' ' * (width - index_pos)
        return result

    def lines(self, width, count_broken=False):
        result = list()
        num_broken = 0
        if self.text is None:
            if count_broken:
                return result, num_broken
            return result

        pos = 0

        while pos < len(self.text):
            if pos > 0:
                # move past line break characters after the first line; they've already been handled
                while pos < len(self.text) \
                        and (self.text[pos] == '\r' or self.text[pos] == '\n'):
                    pos += 1

            if pos >= len(self.text):
                if count_broken:
                    return result, num_broken
                return result

            line_width = width
            current_indent = max(0, self._indent_at_pos(pos))
            if width > current_indent:
                line_width = width - current_indent

            if len(self.text) - pos <= line_width:
                result.append(self._current_line(width, pos, len(self.text)))
                if count_broken:
                    return result, num_broken
                return result

            index_text_break = -1
            for index in range(1, line_width):
                # non-breaking space
                if self.text[pos + index] == 0x00A0:
                    continue
                if self.text[pos + index] == '\r' or self.text[pos + index] == '\n':
                    index_text_break = pos + index
                    break
                elif self.text[pos + index].isspace():
                    index_text_break = pos + index

            if index_text_break < 0:
                index_text_break = pos + line_width
                num_broken += 1

            result.append(self._current_line(width, pos, index_text_break))
            pos = index_text_break

        if count_broken:
            return result, num_broken
        return result

    def set_style(self, indent, style):
        """
        Set the line style.
        :param indent: The number of characters to use for each indentation.
        :param style: The line style to use, one of TextWrapManager.Style settings.
        """
        if indent > 0 and not TextWrapStyle.is_indent_style(style):
            raise ValueError('When indent > 0, style must be an indent style')
        if indent == self._indent and style == self._style:
            return
        self._indent = indent
        self._style = style
        if self._line_counts_at_width is not None:
            self._compute_line_counts()

    def no_word_break_width(self):
        """
        The minimum width at which no words will be broken across lines
        """
        if self._line_counts_at_width is None:
            self._compute_line_counts()
        for width in range(len(self._broken_words_at_width) - 1, 0, -1):
            if self._broken_words_at_width[width] > 0:
                return width + 1
        return 1

    def get_critical_widths(self):
        """
        Gets a list of the widths at which the number of lines for the current text and line style changes.
        :return A list of CriticalWidth instances containing width/line_count tuples.
        """
        if self.text is None or len(self.text) == 0:
            return None
        result = list()
        for width in range(1, len(self.text)):
            line_count = self.line_count_at_width(width)
            last_line_count = -1
            if len(result) > 0:
                last_line_count = result[-1].line_count
            if line_count != last_line_count:
                result.append(CriticalWidth(width=width, line_count=line_count))
        return result

    def line_count_at_width(self, width):
        """
        The number of lines of text this instance would produce at the given width.
        :param width: The width for which to calculate the number of lines.
        :return: The number of lines.
        """
        if self._line_counts_at_width is None:
            self._compute_line_counts()
        if width < 1 and self.text is not None and len(self.text) > 0:
            raise Exception('width must be at least 1')
        if width >= len(self._line_counts_at_width):
            return self._line_counts_at_width[-1]
        return self._line_counts_at_width[width]

    def broken_words_at_width(self, width):
        """
        The number of broken words this instance would produce at the given width.
        :param width:  The width for which to calculate the number of broken words.
        :return: The number of broken words.
        """
        if self._line_counts_at_width is None:
            self._compute_line_counts()
        if width < 1:
            raise Exception('width must be at least 1')
        if width >= len(self._broken_words_at_width):
            return self._broken_words_at_width[-1]
        return self._broken_words_at_width[width]

    def _compute_line_counts(self):
        self._line_counts_at_width = list()
        self._broken_words_at_width = list()
        if self.text is not None:
            for width in range(1, len(self.text) + 1):
                lines, num_broken = self.lines(width, count_broken=True)
                self._line_counts_at_width.append(len(lines))
                self._broken_words_at_width.append(num_broken)
        if len(self._line_counts_at_width) == 0:
            self._line_counts_at_width.append(1)
            self._broken_words_at_width.append(0)
