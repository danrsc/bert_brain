from . import data_loader
from . import university_college_london_corpus

from .university_college_london_corpus import *
from .data_loader import *

__all__ = ['data_loader', 'university_college_london_corpus']
__all__.extend(university_college_london_corpus.__all__)
__all__.extend(data_loader.__all__)
