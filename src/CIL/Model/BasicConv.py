from .BatchNorm2d import BatchNorm2d
from .Conv2d import Conv2d
import torch
from torch import nn
from typing import Tuple, List, Dict, Any, Optional

class BasicConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        kernel_size,
        activation,
        ):
        super().__init__()
        self.conv = Conv2d(
            in_channels= in_channels,
            out_channels= out_channels,
            kernel_size= kernel_size,
            stride= stride,
            padding = kernel_size // 2,
            bias = False,
        )
        # self.bn = BatchNorm2d(
        #     num_features= out_channels,
        #     track_running_stats=False,
        # )
        self.activation = activation
        self.inspect = False
        self.latents = torch.empty(0,in_channels).to("cuda")


    def forward(
        self,
        input
        ):
        output = self.conv(input)
        if self.inspect:
            latent = input.mean(dim=(-1,-2))
            self.latents = torch.cat((self.latents, torch.unsqueeze(latent[0,:],dim=0)),dim=0)
        # output = self.bn(output)
        if self.activation != None:
            output = self.activation(output)
        return output

    def unset_mask(
        self
        ):
        self.conv.unset_mask()
        # self.bn.unset_mask()

    def clear_unused_parameters(
        self,
        effective_input_range_percentage: Tuple[float,float],
        effective_output_range_percentage:Tuple[float,float],
        ):
        input_start, input_end = effective_input_range_percentage
        self.conv.clear_rectange_weight(
            input_range_percentage=(input_end,1),
            output_range_percentage=effective_output_range_percentage,
        )

    def freeze_old_unused_parameters(
        self,
        effective_input_range_percentage: Tuple[float,float],
        effective_output_range_percentage:Tuple[float,float],
        ):
        """
        freeze parameters for old tasks and unseen tasks
        """
        input_start, input_end = effective_input_range_percentage
        output_start ,output_end = effective_output_range_percentage

        self.conv.freeze_rectangle_backward(
            input_range_percentage=(.0,1.),
            output_range_percentage=(.0,output_start),
        )
        self.conv.freeze_rectangle_backward(
            input_range_percentage=(input_end,1.),
            output_range_percentage=effective_output_range_percentage,
        )
        self.conv.freeze_rectangle_backward(
            input_range_percentage=(.0,1.),
            output_range_percentage=(output_end,1.),
        )
        # self.bn.freeze_range_backward(
        #     output_range_percentage=(.0,output_start),
        # )
        # self.bn.freeze_range_backward(
        #     output_range_percentage=(output_end,1.),
        # )
