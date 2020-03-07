from torch.nn import Module


__all__ = ['GraphPart']


class GraphPart(Module):

    def resolve_placeholders(self, placeholder_name_to_fields, field_shapes, num_response_data_fields):
        raise NotImplementedError('{} does not implement resolve_placeholders'.format(type(self)))

    def forward(self, batch):
        raise NotImplementedError('{} does not implement forward'.format(type(self)))

    def instantiate(self, name_to_num_channels):
        raise NotImplementedError('{} does not implement instantiate'.format(type(self)))

    def compute_penalties(self, batch, predictions, loss_dict):
        return None
