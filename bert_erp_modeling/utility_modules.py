import numpy as np
import torch


__all__ = ['GroupPool', 'Conv1DCausal']


class GroupPool(torch.nn.Module):

    def forward(self, x, groupby):

        # first attach an example_id to the groups to ensure that we don't pool across examples in the batch

        # array of shape (batch, sequence, 1) which identifies example
        example_ids = torch.arange(
            groupby.size()[0], device=x.device).view((groupby.size()[0], 1, 1)).repeat((1, groupby.size()[1], 1))
        # -> (batch, sequence, 2): attach example_id to each group
        groupby = torch.cat((groupby.view(groupby.size() + (1,)), example_ids), dim=2)

        # -> (batch * sequence, 2)
        groupby = groupby.view((groupby.size()[0] * groupby.size()[1], groupby.size()[2]))

        # each group is a (group, example_id) tuple
        groups, group_indices = torch.unique(groupby, return_inverse=True, dim=0)

        # split the groups into the true groups and example_ids
        example_ids = groups[:, 1]
        groups = groups[:, 0]

        # -> (batch * sequence, 1, 1, ..., 1)
        group_indices = group_indices.view((x.size()[0] * x.size()[1],) + (1,) * (len(x.size()) - 2))

        # -> (batch * sequence, n, m, ..., k)
        group_indices = group_indices.repeat((1,) + x.size()[2:])

        # -> (batch * sequence, n, m, ..., k)
        x = x.view((x.size()[0] * x.size()[1],) + x.size()[2:])

        pooled = torch.zeros((groups.size()[0],) + x.size()[1:], dtype=x.dtype, device=x.device)
        pooled.scatter_add_(dim=0, index=group_indices, src=x)

        # -> (batch * sequence)
        group_indices = group_indices[:, 0]
        counts = torch.zeros(groups.size()[0], dtype=x.dtype, device=x.device)
        counts.scatter_add_(
            dim=0, index=group_indices, src=torch.ones(len(group_indices), dtype=x.dtype, device=x.device))
        counts = counts.view(counts.size() + (1,) * len(pooled.size()[1:]))
        pooled = pooled / counts

        # filter out groups < 0
        indicator_valid = groups >= 0
        pooled = pooled[indicator_valid]
        groups = groups[indicator_valid]
        example_ids = example_ids[indicator_valid]

        return pooled, groups, example_ids


class Conv1DCausal(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True,
                 transpose_axes=None, should_transpose_input=True, should_transpose_output=True):
        super().__init__()
        self.transpose_axes = transpose_axes
        self.should_transpose_input = should_transpose_input
        self.should_transpose_output = should_transpose_output
        padding = dilation * (kernel_size - 1)
        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        if self.transpose_axes is not None and self.should_transpose_input:
            x = x.permute(*self.transpose_axes)
        result = self.conv1d(x)
        # remove the element from the right padding
        result = result[:, :, :-self.conv1d.padding[0]]
        if self.transpose_axes is not None and self.should_transpose_output:
            result = result.permute(*np.argsort(self.transpose_axes))
        return result
