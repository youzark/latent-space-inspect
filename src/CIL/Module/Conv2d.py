import torch
from torch import nn
from typing import Union
from torch.nn.common_types import _size_2_t
from typing import Tuple,List, Dict

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
        self.hooks = []
        self.store_latents = False
        self.in_latents = torch.empty(0,in_channels)
        self.out_latents = torch.empty(0,out_channels)

    def forward(self, input):
        output = super().forward(input)
        with torch.no_grad():
            if self.store_latents:
                self.in_latents = torch.stack((self.in_latents,input.detach().view(-1,self.in_channels)))
                self.out_latents = torch.stack((self.out_latents,output.detach().view(-1,self.out_channels)))
        return output

    def rearrange_output(self, index):
        with torch.no_grad():
            weight = self.weight
            weight[range(self.out_channels),:,:,:] = weight[index,:,:,:]

    def rearrange_input(self, index):
        with torch.no_grad():
            weight = self.weight
            weight[:,range(self.in_channels),:,:] = weight[:,index,:,:]

    def freeze_rectangle_backward(
        self,
        input_range_percentage: Tuple[float,float],
        output_range_percentage:Tuple[float,float]
        ):
        in_start_per, in_end_per = input_range_percentage
        out_start_per, out_end_per = output_range_percentage
        in_start, out_start = (int)(in_start_per * self.in_channels), (int)(out_start_per * self.out_channels)
        in_end, out_end = (int)(in_end_per * self.in_channels), (int)(out_end_per * self.out_channels)
        def weight_mask(grad):
            with torch.no_grad():
                grad[out_start:out_end,in_start:in_end,:,:] *= 0.
            return grad
        def bias_mask(grad):
            with torch.no_grad():
                grad[out_start:out_end] *= 0
            return grad
        self.hooks.append(self.weight.register_hook(weight_mask))
        if self.bias != None:
            self.hooks.append(self.bias.register_hook(bias_mask))

    def clear_rectange_weight(
        self,
        input_range_percentage: Tuple[float,float],
        output_range_percentage:Tuple[float,float],
        clear_bias=False
        ):
        in_start_per, in_end_per = input_range_percentage
        out_start_per, out_end_per = output_range_percentage
        in_start, out_start = (int)(in_start_per * self.in_channels), (int)(out_start_per * self.out_channels)
        in_end, out_end = (int)(in_end_per * self.in_channels), (int)(out_end_per * self.out_channels)
        with torch.no_grad():
            self.weight[out_start:out_end,in_start:in_end,:,:] *= 0
            if clear_bias and self.bias != None:
                self.bias[out_start:out_end] *= 0

    def unset_mask(
        self
        ):
        for hook in self.hooks:
            hook.remove()

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
