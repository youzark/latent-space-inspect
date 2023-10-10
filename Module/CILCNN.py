from .BatchNorm2d import BatchNorm2d
from .Conv2d import Conv2d
from .Linear import Linear
import torch
from torch import nn
from typing import Tuple, List, Dict, Any, Optional
import torch.nn.utils.weight_norm as WeightNorm

class CILCNN(nn.Module):
    def __init__(
        self,
        ):
        super().__init__()
        self.activation = nn.ReLU()
        self.block1 = BasicConv(
            in_channels = 1,
            out_channels = 10,
            stride=2,
            kernel_size=3,
            activation=self.activation
        )
        self.block2 = BasicConv(
            in_channels = 10,
            out_channels = 20,
            stride=2,
            kernel_size=3,
            activation=self.activation
        )
        self.block3 = BasicConv(
            in_channels = 20,
            out_channels = 40,
            stride=2,
            kernel_size=3,
            activation=self.activation
        )

        self.linear = Linear(
            in_features=40*8*8,
            out_features=10,
            bias=False,
        )


        self.layers = nn.Sequential(
            self.block1,
            self.block2,
            self.block3,
        )

        self.softmax = nn.Softmax(dim=1)

        self.inspect = False
        self.latents = torch.empty(0,40).to("cuda")

    def forward(
        self,
        input
        ):
        temp = self.layers(input)
        if self.inspect:
            latent = temp.mean(dim=(-1,-2))
            self.latents = torch.cat((self.latents, torch.unsqueeze(latent[0,:],dim=0)),dim=0)

        logits = self.linear(temp.view(temp.shape[0],-1))
        return self.softmax(logits)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def unfreeze(
        self
        ):
        self.block1.unset_mask()
        self.block2.unset_mask()
        self.block3.unset_mask()
        self.linear.unset_mask()

    def freeze_backward(
        self,
        effective_parameter_percentage: Tuple[float,float],
        ):
        """
        freeze parameters for old tasks and unseen tasks
        """
        start, end = effective_parameter_percentage
        self.block1.freeze_old_unused_parameters(
            effective_input_range_percentage=(0.,1.),
            effective_output_range_percentage=effective_parameter_percentage
        )

        self.block2.freeze_old_unused_parameters(
            effective_input_range_percentage=effective_parameter_percentage,
            effective_output_range_percentage=effective_parameter_percentage
        )
        self.block2.clear_unused_parameters(
            effective_input_range_percentage=effective_parameter_percentage,
            effective_output_range_percentage=effective_parameter_percentage
        )

        self.block3.freeze_old_unused_parameters(
            effective_input_range_percentage=effective_parameter_percentage,
            effective_output_range_percentage=effective_parameter_percentage
        )
        self.block3.clear_unused_parameters(
            effective_input_range_percentage=effective_parameter_percentage,
            effective_output_range_percentage=effective_parameter_percentage
        )

        self.linear.freeze_rectangle_backward(
            input_range_percentage=(0,start),
            output_range_percentage=(0,1)
        )
        self.linear.freeze_rectangle_backward(
            input_range_percentage=(end,1),
            output_range_percentage=(0,1)
        )


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



class Downsample(nn.Module):
    def __init__(
        self,
        ):
        pass

class BottleNeckStack(nn.Module):
    def __init__(
        self,
        ):
        pass

class BottleNeck(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        down_scaling_factor,
        stride,
        kernel,
        activation,
        ):
        mid_channels = in_channels // down_scaling_factor
        pass
