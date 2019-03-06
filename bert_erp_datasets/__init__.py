from . import colorless_green
from . import data_loader
from . import data_preparer
from . import dataset
from . import harry_potter
from . import natural_stories
from . import university_college_london_corpus

from .colorless_green import *
from .data_loader import *
from .data_preparer import *
from .dataset import *
from .harry_potter import *
from .natural_stories import *
from .university_college_london_corpus import *


__all__ = [
    'colorless_green', 'data_loader', 'data_preparer', 'dataset', 'harry_potter', 'natural_stories',
    'university_college_london_corpus']
__all__.extend(colorless_green.__all__)
__all__.extend(data_loader.__all__)
__all__.extend(data_preparer.__all__)
__all__.extend(dataset.__all__)
__all__.extend(harry_potter.__all__)
__all__.extend(natural_stories.__all__)
__all__.extend(university_college_london_corpus.__all__)
