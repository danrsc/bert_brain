from . import critic_types
from . import keyed_modules
from . import graph_part
from . import grouping_modules
from . import bert_multi_prediction_head
from . import contextual_parameter_generation
from . import specialized_loss
from . import utility_modules

from .critic_types import *
from .keyed_modules import *
from .graph_part import *
from .grouping_modules import *
from .bert_multi_prediction_head import *
from .contextual_parameter_generation import *
from .specialized_loss import *
from .utility_modules import *

__all__ = ['critic_types', 'keyed_modules', 'graph_part', 'grouping_modules', 'bert_multi_prediction_head',
           'specialized_loss', 'utility_modules', 'contextual_parameter_generation']
__all__.extend(keyed_modules.__all__)
__all__.extend(graph_part.__all__)
__all__.extend(grouping_modules.__all__)
__all__.extend(bert_multi_prediction_head.__all__)
__all__.extend(contextual_parameter_generation.__all__)
__all__.extend(specialized_loss.__all__)
__all__.extend(utility_modules.__all__)


def _assert_critic_subclasses_recursive():
    def sub(c, result):
        if not c.__name__.startswith('_'):
            result.add(c.__name__)
        for sc in c.__subclasses__():
            sub(sc, result)
    all_critic_types = set()
    for cb in NamedTargetMaskedLossBase.__subclasses__():
        sub(cb, all_critic_types)

    registered_critic_types = set(critic_types.__all__)
    unregistered_types = all_critic_types - registered_critic_types
    if len(unregistered_types) > 0:
        raise AssertionError('Some critics are not registered in critic_types: {}'.format(unregistered_types))


_assert_critic_subclasses_recursive()
