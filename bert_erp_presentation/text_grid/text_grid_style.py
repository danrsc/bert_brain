__all__ = ['TextGridStyle', 'EmptyStyleHandler']


class TextGridStyle:
    """
    Class used to define how parts of a table should be rendered.
    """

    def __init__(
            self,
            row_padding=-1,
            column_padding=-1,
            width=-1,
            left_margin=-1,
            right_margin=-1,
            border=None,
            background_color=None,
            text_color=None,
            bold=None,
            line_style=None,
            line_indent=-1,
            additional_style_attribute_dictionary=None,
            parent_style=None):
        self._row_padding = row_padding
        self._column_padding = column_padding
        self._width = width
        self._left_margin = left_margin
        self._right_margin = right_margin
        self._border = border
        self._background_color = background_color
        self._text_color = text_color
        self._bold = bold
        self._line_style = line_style
        self._line_indent = line_indent
        self._additional_style_attribute_dictionary = additional_style_attribute_dictionary
        self.parent_style = parent_style

    def copy(self):

        additional_style_attributes = None
        if self._additional_style_attribute_dictionary is not None:
            additional_style_attributes = dict(self._additional_style_attribute_dictionary)

        return TextGridStyle(
            row_padding=self._row_padding,
            column_padding=self._column_padding,
            width=self._width,
            left_margin=self._left_margin,
            right_margin=self._right_margin,
            border=self._border,
            background_color=self._background_color,
            text_color=self._text_color,
            bold=self._bold,
            line_style=self._line_style,
            line_indent=self._line_indent,
            additional_style_attribute_dictionary=additional_style_attributes,
            parent_style=self.parent_style)

    def _get_style(self, style_name, default_value):
        local_value = getattr(self, style_name)
        if (default_value is None and local_value is None) or local_value == default_value:
            if self.parent_style is None:
                return local_value
            # noinspection PyProtectedMember
            return self.parent_style._get_style(style_name)
        return local_value

    def get_row_padding(self):
        return self._get_style('_row_padding', -1)

    def get_column_padding(self):
        return self._get_style('_column_padding', -1)

    def get_width(self):
        return self._get_style('_width', -1)

    def get_left_margin(self):
        return self._get_style('_left_margin', -1)

    def get_right_margin(self):
        return self._get_style('_right_margin', -1)

    def get_border(self):
        return self._get_style('_border', -1)

    def get_background_color(self):
        return self._get_style('_background_color', None)

    def get_text_color(self):
        return self._get_style('_text_color', None)

    def get_bold(self):
        return self._get_style('_bold', None)

    def get_line_style(self):
        return self._get_style('_line_style', None)

    def get_line_indent(self):
        return self._get_style('_line_indent', -1)

    def get_additional_style_attributes(self, attributes=None):
        if self.parent_style is not None:
            result = self.parent_style.get_additional_style_attributes(attributes)
        else:
            result = dict()

        if self._additional_style_attribute_dictionary is None:
            return result

        if attributes is not None:
            if isinstance(attributes, str):
                attributes = [attributes]
            for attribute in attributes:
                if attribute in self._additional_style_attribute_dictionary:
                    result[attribute] = self._additional_style_attribute_dictionary[attribute]
        else:
            for attribute in self._additional_style_attribute_dictionary.iterkeys():
                result[attribute] = self._additional_style_attribute_dictionary[attribute]
        return result

    def get_parent_style(self):
        return self.parent_style

    def set_row_padding(self, value):
        self._row_padding = value
        return self

    def set_column_padding(self, value):
        self._column_padding = value
        return self

    def set_width(self, value):
        self._width = value
        return self

    def set_left_margin(self, value):
        self._left_margin = value
        return self

    def set_right_margin(self, value):
        self._right_margin = value
        return self

    def set_border(self, value):
        self._border = value
        return self

    def set_background_color(self, value):
        self._background_color = value
        return self

    def set_text_color(self, value):
        self._text_color = value
        return self

    def set_bold(self, value):
        self._bold = value
        return self

    def set_line_style(self, value):
        self._line_style = value
        return self

    def set_line_indent(self, value):
        self._line_indent = value
        return self

    def set_additional_style_attribute(self, attribute, value):
        if self._additional_style_attribute_dictionary is None:
            self._additional_style_attribute_dictionary = dict()
        self._additional_style_attribute_dictionary[attribute] = value
        return self

    def set_parent_style(self, value):
        self.parent_style = value
        return self


class EmptyStyleHandler:
    """
    A style handler that doesn't do any styling.
    """

    def __init__(self):
        self.count = 0

    def push_style(self, style):
        if style is None:
            pass
        self.count += 1

    def pop_style(self):
        if self.count <= 0:
            raise Exception('no styles to pop')
        self.count -= 1
