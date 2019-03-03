from . import data_loader
from . import data_preparer
from . import dataset
from . import harry_potter
from . import natural_stories
from . import number_dataset
from . import university_college_london_corpus

from .data_loader import *
from .data_preparer import *
from .dataset import *
from .harry_potter import *
from .natural_stories import *
from .number_dataset import *
from .university_college_london_corpus import *


__all__ = [
    'data_loader', 'data_preparer', 'dataset', 'harry_potter', 'natural_stories', 'number_dataset',
    'university_college_london_corpus']
__all__.extend(data_loader.__all__)
__all__.extend(data_preparer.__all__)
__all__.extend(dataset.__all__)
__all__.extend(harry_potter.__all__)
__all__.extend(natural_stories.__all__)
__all__.extend(number_dataset.__all__)
__all__.extend(university_college_london_corpus.__all__)
