from .text_cell import TextCell
from .text_grid_style import TextGridStyle


__all__ = ['TextGrid']


class _TextGridSpec:
    """
    Class to be used internally within the package. Do not construct directly
    """

    def __init__(
            self, value: TextCell, index_grid_left: int, index_grid_top: int, num_rows: int = 1, num_columns: int = 1):
        self._value: TextCell = value
        self._index_grid_left: int = index_grid_left
        self._index_grid_top: int = index_grid_top
        self._num_rows: int = num_rows
        self._num_columns: int = num_columns

    @property
    def value(self):
        return self._value

    @property
    def index_grid_left(self):
        return self._index_grid_left

    @property
    def index_grid_top(self):
        return self._index_grid_top

    @property
    def num_rows(self):
        return self._num_rows

    @property
    def num_columns(self):
        return self._num_columns

    def update(self, index_grid_left=None, index_grid_top=None, num_rows=None, num_columns=None):
        if index_grid_left is not None:
            self._index_grid_left = index_grid_left
        if index_grid_top is not None:
            self._index_grid_top = index_grid_top
        if num_rows is not None:
            self._num_rows = num_rows
        if num_columns is not None:
            self._num_columns = num_columns


class _RemoveChoice:
    def __init__(self):
        pass

    top = 'top'
    bottom = 'bottom'
    left = 'left'
    right = 'right'
    all = 'all'


class TextGrid:

    def __init__(self, items=None, num_rows=None, num_columns=None, **text_cell_styles):
        self._current_row = 0
        self._current_column = 0
        self._num_fixed_columns = -1
        self._num_fixed_rows = -1
        self._grid = []
        self._specs = list()

        if num_rows is not None:
            if num_rows >= 1:
                self._num_fixed_rows = num_rows
        if num_columns is not None:
            if num_columns >= 1:
                self._num_fixed_columns = num_columns

        if items is not None:
            items = list(items)
            if len(items) > 0:
                has_iterable = False
                has_value = False
                for item in items:
                    if isinstance(item, str) or isinstance(item, TextCell):
                        has_value = True
                    else:
                        has_iterable = True
                if has_value and has_iterable:
                    raise ValueError('Cannot mix iterables and strings in iterable')
                if has_value:
                    for item in items:
                        self.append_value(item, **text_cell_styles)
                else:
                    for index_row, row in enumerate(items):
                        for item in row:
                            self.append_value(item, **text_cell_styles)
                        if index_row < len(items) - 1:
                            self.next_row()

    @property
    def row_count(self):
        if self._grid is None:
            return 0
        return len(self._grid)

    @property
    def column_count(self):
        if self._grid is None or len(self._grid) == 0:
            return 0
        return len(self._grid[0])

    @property
    def current_row(self):
        return self._current_row

    @property
    def current_column(self):
        return self._current_column

    def _check_grid(self):
        if len(self._grid) > 0 and len(self._grid[-1]) != len(self._grid[0]):
            raise RuntimeError('wrong number of columns in current row: {0}'.format(len(self._grid) - 1))

    @staticmethod
    def _is_spec(x):
        return (
            (isinstance(x, tuple) or isinstance(x, list))
            and len(x) == 3
            and isinstance(x[1], int)
            and isinstance(x[2], int))

    def append_value(self, value, **text_cell_styles):

        if TextGrid._is_spec(value):
            value, num_rows, num_columns = value
        else:
            num_rows = 1
            num_columns = 1

        if isinstance(value, str):
            value = TextCell(value, TextGridStyle(**text_cell_styles))

        # noinspection PyTypeChecker
        if 0 <= self._num_fixed_columns < self.current_column + num_columns:
            # noinspection PyTypeChecker
            if num_columns > self._num_fixed_columns:
                raise ValueError('value out of bounds for fixed number of columns')
            # noinspection PyTypeChecker
            if 0 <= self._num_fixed_rows <= self.current_row + 1:
                raise ValueError('new row is required, but would be out of bounds')
            self.next_row()

        # noinspection PyTypeChecker
        if 0 <= self._num_fixed_rows < self.current_row + num_rows:
            raise ValueError('value out of bounds for fixed number of rows')

        value_spec = None
        for i in range(self.current_row, self.current_row + num_rows):
            if i == len(self._grid):
                self._grid.append(list())
            if i == self.current_row:
                while self.current_column < len(self._grid[i]) and \
                        self._grid[i][self.current_column] is not None:
                    self._current_column += 1
                value_spec = _TextGridSpec(
                    value,
                    index_grid_left=self.current_column,
                    index_grid_top=self.current_row,
                    num_rows=num_rows,
                    num_columns=num_columns)
            else:
                while len(self._grid[i]) < self.current_column:
                    self._grid[i].append(None)
            for j in range(self.current_column, self.current_column + num_columns):
                if j == len(self._grid[i]):
                    self._grid[i].append(value_spec)
                elif self._grid[i][j] is None:
                    self._grid[i][j] = value_spec
                else:
                    raise Exception('bad grid')

        self._specs.append(value_spec)
        self._current_column += value_spec.num_columns

    def next_row(self):

        # noinspection PyTypeChecker
        if 0 <= self._num_fixed_rows <= self.current_row + 1:
            raise RuntimeError('Adding a row would exceed the number of fixed rows')

        if self.current_row < len(self._grid):
            while self.current_column < len(self._grid[self.current_row]) \
                    and self._grid[self.current_row][self.current_column] is not None:
                self._current_column += 1
        if len(self._grid) > 0 and self.current_column != len(self._grid[0]):
            raise RuntimeError('wrong number of columns in current row: {0}, expected {1}, got {2}'.format(
                self.current_row, len(self._grid[0]), self.current_column))
        self._current_column = 0
        self._current_row += 1

    def fix_rows(self):
        self._num_fixed_rows = self.row_count

    def fix_columns(self):
        self._num_fixed_columns = self.column_count

    @staticmethod
    def remove_choice(new_spec, value_spec):
        if value_spec.index_grid_top < new_spec.index_grid_top:
            if value_spec.index_grid_left < new_spec.index_grid_left:
                # we've lost the lower right corner
                if (new_spec.index_grid_left - value_spec.index_grid_left >
                        new_spec.index_grid_top - value_spec.index_grid_top):
                    return _RemoveChoice.right
                else:
                    # if equal, prefer to keep horizontal space
                    return _RemoveChoice.bottom
            elif (new_spec.index_grid_left + new_spec.num_columns <
                  value_spec.index_grid_left + value_spec.num_columns):
                # we've lost the lower left corner
                if (value_spec.index_grid_left + value_spec.num_columns -
                        (new_spec.index_grid_left + new_spec.num_columns) >
                        new_spec.index_grid_top - value_spec.index_grid_top):
                    return _RemoveChoice.left
                else:
                    # if equal, prefer to keep horizontal space
                    return _RemoveChoice.bottom
            else:
                # we've lost the bottom
                return _RemoveChoice.bottom
        elif new_spec.index_grid_top + new_spec.num_rows < value_spec.index_grid_top + value_spec.num_rows:
            if value_spec.index_grid_left < new_spec.index_grid_left:
                # we've lost the top right corner
                if (new_spec.index_grid_left - value_spec.index_grid_left >
                        value_spec.index_grid_top + value_spec.num_rows -
                        (new_spec.index_grid_top + new_spec.num_rows)):
                    return _RemoveChoice.right
                else:
                    return _RemoveChoice.top
            elif (new_spec.index_grid_left + new_spec.num_columns <
                  value_spec.index_grid_left + value_spec.num_columns):
                # we've lost the top left corner
                if (value_spec.index_grid_left + value_spec.num_columns -
                        (new_spec.index_grid_left + new_spec.num_columns) >
                        new_spec.index_grid_top - value_spec.index_grid_top):
                    return _RemoveChoice.left
                else:
                    # if equal, prefer to keep horizontal space
                    return _RemoveChoice.top
            else:
                # we've lost the top
                return _RemoveChoice.top
        elif value_spec.index_grid_left < new_spec.index_grid_left:
            # we've lost the right
            return _RemoveChoice.right
        elif (new_spec.index_grid_left + new_spec.num_columns <
              value_spec.index_grid_left + value_spec.num_columns):
            # we've lost the left
            return _RemoveChoice.left
        else:
            return _RemoveChoice.all

    def __setitem__(self, key, value):

        if not isinstance(key, tuple):
            raise IndexError('key must be an (index_row, index_column) pair')
        try:
            row_key, column_key = key
        except ValueError:
            raise ValueError('key must be an (index_row, index_column) pair')

        if isinstance(value, str):
            value = TextCell(value)
        if not isinstance(value, TextCell):
            raise ValueError('Unexpected value type: {}'.format(type(value)))

        if isinstance(row_key, slice):
            row1, row2, row_step = row_key.indices(max(row_key.start, row_key.stop, self.row_count))
            if row_step != 1:
                raise IndexError('row index cannot use step other than 1')
        else:
            if row_key < 0:
                row_key += self.row_count
            if row_key < 0:
                raise IndexError('row index out of range: {0}'.format(row_key))
            row1, row2 = row_key, row_key + 1

        if isinstance(column_key, slice):
            col1, col2, column_step = column_key.indices(max(column_key.start, column_key.stop, self.column_count))
            if column_step != 1:
                raise IndexError('column index cannot use step other than 1')
        else:
            if column_key < 0:
                column_key += self.column_count
            if column_key < 0:
                raise IndexError('column index out of range: {0}'.format(column_key))
            col1, col2 = column_key, column_key + 1

        # noinspection PyTypeChecker
        if 0 <= self._num_fixed_columns < col2:
            raise IndexError('column index out of range: {0}'.format(column_key))
        # noinspection PyTypeChecker
        if 0 <= self._num_fixed_rows < row2:
            raise IndexError('row index out of range: {0}'.format(row_key))

        if self.column_count < col2:
            for i in range(self.row_count):
                while len(self._grid[i]) < col2:
                    self._grid[i].append(None)

        while self.row_count < row2:
            self._grid.append([None] * max(self.column_count, col2))

        new_spec = _TextGridSpec(value, col1, row1, row2 - row1, col2 - col1)
        self._specs.append(new_spec)

        to_modify = dict()

        for index_row in range(row1, row2):
            for index_column in range(col1, col2):
                value_spec = self._grid[index_row][index_column]
                if value_spec is not None:
                    to_modify[(value_spec.index_grid_top, value_spec.index_grid_left)] = value_spec

        for value_spec in to_modify.values():
            remove_choice = TextGrid.remove_choice(new_spec, value_spec)
            start_row, end_row = value_spec.index_grid_top, value_spec.index_grid_top + value_spec.num_rows
            start_column, end_column = value_spec.index_grid_left, value_spec.index_grid_left + value_spec.num_columns
            if remove_choice == _RemoveChoice.right:
                start_column = new_spec.index_grid_left
                value_spec.update(num_columns=new_spec.index_grid_left - value_spec.index_grid_left)
            elif remove_choice == _RemoveChoice.bottom:
                start_row = new_spec.index_grid_top
                value_spec.update(num_rows=new_spec.index_grid_top - value_spec.index_grid_top)
            elif remove_choice == _RemoveChoice.left:
                new_left = new_spec.index_grid_left + new_spec.num_columns
                end_column = new_left
                value_spec.update(
                    index_grid_left=new_left,
                    num_columns=value_spec.index_grid_left + value_spec.num_columns - new_left)
            elif remove_choice == _RemoveChoice.top:
                new_top = new_spec.index_grid_top + new_spec.num_rows
                end_row = new_top
                value_spec.update(
                    index_grid_top=new_top,
                    num_rows=value_spec.index_grid_top + value_spec.num_rows - new_top)
            elif remove_choice == _RemoveChoice.all:
                self._specs.remove(value_spec)
            else:
                raise RuntimeError('Bad code - unknown remove_choice: {0}'.format(remove_choice))

            for index_row in range(start_row, end_row):
                for index_column in range(start_column, end_column):
                    self._grid[index_row][index_column] = None

        for index_row in range(row1, row2):
            for index_column in range(col1, col2):
                self._grid[index_row][index_column] = new_spec

    def is_part_of_previous_row(self, index_row, index_column):
        """
        Returns True if the element at the specified row and column is spanned by a cell with its origin in a previous
        row. False otherwise
        :param index_row: The index of the desired row.
        :param index_column: The index of the desired column.
        """
        if index_row == self.row_count:
            return False
        if index_column == self.column_count:
            index_column -= 1
        spec = self._grid[index_row][index_column]
        if spec is None:
            return False
        return spec.index_grid_top < index_row

    def is_part_of_previous_column(self, index_row, index_column):
        """
        Returns True if the element at the specified row and column is spanned by a cell with its origin in a previous
        column. False otherwise.
        :param index_row: The index of the desired row.
        :param index_column: The index of the desired column.
        """
        if index_row == self.row_count:
            index_row -= 1
        if index_column == self.column_count:
            return False
        spec = self._grid[index_row][index_column]
        if spec is None:
            return False
        return spec.index_grid_left < index_column

    def get_spec(self, index_row, index_column):
        return self._grid[index_row][index_column]

    def iterate_specs(self):
        for spec in self._specs:
            yield spec
