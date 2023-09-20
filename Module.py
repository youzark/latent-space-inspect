import torch
from torch import nn
from typing import Union
from torch.nn.common_types import _size_2_t

class Conv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None

        ):
        super().__init__(
            in_channels= in_channels,
            out_channels= out_channels,
            kernel_size = kernel_size,
            stride= stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )

    def rearrange_output(self, index):
        with torch.no_grad():
            weight = self.weight
            weight[range(self.out_channels),:,:,:] = weight[index,:,:,:]

    def rearrange_input(self, index):
        with torch.no_grad():
            weight = self.weight
            weight[:,range(self.in_channels),:,:] = weight[:,index,:,:]

    def truncate_output(
            self,
            num_trunc:int,
            trunc_tail:bool=True, #if False, truncate from low dimension
        ):
        with torch.no_grad():
            in_channels = self.in_channels
            out_channels = self.out_channels
            kernel_size = self.kernel_size
            padding = self.padding
            stride = self.stride
            bias = self.bias

            truncated_old_weight = self.conv.weight[:-num_trunc,:,:,:] if trunc_tail else self.conv.weight[num_trunc:,:,:,:]

            truncated_old_bias=None
            if self.bias != None:
                truncated_old_bias = self.bias[:-num_trunc] if trunc_tail else self.bias[num_trunc:]

            self.conv = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels - num_trunc,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=bias
            )

            self.conv.weight = nn.Parameter(truncated_old_weight)
            if truncated_old_bias != None:
                self.conv.bias = nn.Parameter(truncated_old_bias)

    def truncate_input(
            self,
            num_trunc:int,
            trunc_tail:bool=True, #if False, truncate from low dimension
        ):
        with torch.no_grad():
            in_channels = self.conv.in_channels
            out_channels = self.conv.out_channels
            kernel_size = self.conv.kernel_size
            padding = self.conv.padding
            stride = self.conv.stride
            bias = self.conv.bias

            truncated_old_weight = self.conv.weight[:,:-num_trunc,:,:] if trunc_tail else self.conv.weight[:,num_trunc:,:,:]

            self.conv = nn.Conv2d(
                in_channels=in_channels - num_trunc,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=bias
            )

            self.conv.weight = nn.Parameter(truncated_old_weight)
