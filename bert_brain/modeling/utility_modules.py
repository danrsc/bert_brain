from collections import OrderedDict
import numpy as np
import torch
import torch.nn


__all__ = ['Conv1DCausal']


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
