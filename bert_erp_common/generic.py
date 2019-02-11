import inspect
from itertools import zip_longest
from _collections_abc import Mapping


__all__ = ['zip_equal', 'copy_from_properties', 'get_keyword_properties', 'FrozenCopyOfDict', 'SwitchRemember']


def zip_equal(*it):
    """
    Like zip, but raises a ValueError if the iterables are not of equal length
    Args:
        *it: The iterables to zip

    Returns:
        yields a tuple of items, one from each iterable
    """
    # wrap the iterators in an enumerate to guarantee that None is a legitimate sentinel
    iterators = [enumerate(i) for i in it]
    for idx, item in enumerate(zip_longest(*iterators)):
        try:
            result = tuple(part[1] for part in item)
            yield result
        except TypeError:
            culprit = None
            for idx_part, part in enumerate(item):
                if part is None:
                    culprit = idx_part
                    break
            raise ValueError(
                'Unequal number of elements in iterators. Problem occurred at index: {}, iterator_index: {}'.format(
                    idx, culprit))


def copy_from_properties(instance, **kwargs):
    """
    Returns a copy of instance by calling __init__ with keyword arguments matching the properties of instance.
    The values of these keyword arguments are taken from the properties of instance except where overridden by
    kwargs. Thus for a class Foo with properties [a, b, c], copy_from_properties(instance, a=7) is equivalent to
    Foo(a=7, b=instance.b, c=instance.c)
    Args:
        instance: The instance to use as a template
        **kwargs: The keyword arguments to __init__ that should not come from the current instance's properties

    Returns:
        A copy of instance modified according to kwargs
    """
    property_names = [n for n, v in inspect.getmembers(type(instance), lambda m: isinstance(m, property))]
    init_kwargs = inspect.getfullargspec(type(instance).__init__).args

    def __iterate_key_values():
        for k in init_kwargs[1:]:
            if k in kwargs:
                yield k, kwargs[k]
            elif k in property_names:
                yield k, getattr(instance, k)

    return type(instance)(**dict(__iterate_key_values()))


def get_keyword_properties(instance, just_names=False):

    property_names = [n for n, v in inspect.getmembers(type(instance), lambda m: isinstance(m, property))]
    init_kwargs = inspect.getfullargspec(type(instance).__init__).args

    if just_names:
        return [k for k in init_kwargs if k in property_names]

    return [(k, getattr(instance, k)) for k in init_kwargs if k in property_names]


class FrozenCopyOfDict(Mapping):

    def __init__(self, mapping):
        if isinstance(mapping, FrozenCopyOfDict):
            self._dict = type(mapping._dict)(mapping._dict)
        else:
            self._dict = type(mapping)(mapping)

    def __getitem__(self, item):
        return self._dict[item]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    @staticmethod
    def replace(instance, mapping):
        result = FrozenCopyOfDict(instance)
        for k in mapping:
            result._dict[k] = mapping[k]
        return result


class SwitchRemember:

    def __init__(self, var):
        self.var = var
        self._tests = set()

    @property
    def tests(self):
        return list(sorted(self._tests))

    def __eq__(self, test):
        self._tests.add(test)
        return self.var == test
