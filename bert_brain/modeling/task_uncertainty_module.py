from collections import OrderedDict
from dataclasses import dataclass
from torch import nn
from .graph_part import GraphPart, GraphPartFactory


__all__ = ['TaskUncertaintyModule', 'TaskUncertaintyModuleFactory']


@dataclass(frozen=True)
class TaskUncertaintyModuleFactory(GraphPartFactory):
    task_id_source_name: str
    output_name: str
    num_tasks: int

    def make_graph_part(self):
        return TaskUncertaintyModule(self.task_id_source_name, self.output_name, self.num_tasks)


class TaskUncertaintyModule(GraphPart):

    def __init__(
            self,
            task_id_source_name,
            output_name,
            num_tasks):
        super().__init__()
        self.task_id_source_name = task_id_source_name
        self.output_name = output_name
        self.num_tasks = num_tasks
        self.embedding = None

    def resolve_placeholders(self, placeholder_name_to_fields, field_shapes, num_response_data_fields):
        if self.num_tasks == 'num_response_data_fields':
            self.num_tasks = num_response_data_fields

    def instantiate(self, name_to_num_channels):
        self.embedding = nn.Embedding(self.num_tasks, 1)
        result = OrderedDict()
        result[self.output_name] = 1

    def forward(self, batch):
        return self.embedding(batch[self.task_id_source_name])
