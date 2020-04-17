from . import attention_key_values
from . import bert_multi_prediction_head
from . import contextual_parameter_generation
from . import critic_types
from . import graph_part
from . import grouping_modules
from . import keyed_modules
from . import learning_rate_schedule_factories
from . import learning_rate_schedules
from . import multi_layer_bottleneck
from . import pareto_solver
from . import specialized_loss
from . import task_uncertainty_module
from . import utility_modules

from .attention_key_values import *
from .bert_multi_prediction_head import *
from .contextual_parameter_generation import *
from .critic_types import *
from .graph_part import *
from .grouping_modules import *
from .keyed_modules import *
from .learning_rate_schedule_factories import *
from .learning_rate_schedules import *
from .multi_layer_bottleneck import *
from .pareto_solver import *
from .specialized_loss import *
from .task_uncertainty_module import *
from .utility_modules import *

__all__ = [
    'attention_key_values',
    'bert_multi_prediction_head',
    'contextual_parameter_generation',
    'critic_types',
    'graph_part',
    'grouping_modules',
    'keyed_modules',
    'learning_rate_schedule_factories',
    'learning_rate_schedules',
    'multi_layer_bottleneck',
    'pareto_solver',
    'specialized_loss',
    'task_uncertainty_module',
    'utility_modules']
__all__.extend(attention_key_values.__all__)
__all__.extend(bert_multi_prediction_head.__all__)
__all__.extend(contextual_parameter_generation.__all__)
__all__.extend(graph_part.__all__)
__all__.extend(grouping_modules.__all__)
__all__.extend(keyed_modules.__all__)
__all__.extend(learning_rate_schedule_factories.__all__)
__all__.extend(multi_layer_bottleneck.__all__)
__all__.extend(pareto_solver.__all__)
__all__.extend(specialized_loss.__all__)
__all__.extend(task_uncertainty_module.__all__)
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


def _assert_learning_rate_subclasses_recursive():
    def sub(c, result):
        if not c.__name__.startswith('_'):
            result.add(c.__name__)
        for sc in c.__subclasses__():
            sub(sc, result)
    all_lr_types = set()
    for cb in LearningRateScheduleFactory.__subclasses__():
        sub(cb, all_lr_types)

    registered_critic_types = set(learning_rate_schedules.__all__)
    unregistered_types = all_lr_types - registered_critic_types
    if len(unregistered_types) > 0:
        raise AssertionError('Some learning rate schedules are not registered in learning_rate_schedules: {}'.format(
            unregistered_types))


_assert_critic_subclasses_recursive()
_assert_learning_rate_subclasses_recursive()
