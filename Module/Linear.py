import torch
from torch import inference_mode, nn
from typing import Union
from torch.nn.common_types import _size_2_t
from typing import Tuple,List, Dict

class Linear(nn.Linear):
    def __init__(
        self,
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        device=None, 
        dtype=None,
        ):
        super().__init__(
            in_features= in_features,
            out_features= out_features,
            bias= bias,
            device= device,
            dtype= dtype,
        )
        self.hooks = []

    def forward(self, input):
        return super().forward(input)


    def freeze_rectangle_backward(
        self,
        input_range_percentage: Tuple[float,float],
        output_range_percentage:Tuple[float,float]
        ):

        in_start_per, in_end_per = input_range_percentage
        out_start_per, out_end_per = output_range_percentage
        in_start, out_start = (int)(in_start_per * self.in_features), (int)(out_start_per * self.out_features)
        in_end, out_end = (int)(in_end_per * self.in_features), (int)(out_end_per * self.out_features)

        def weight_mask(grad):
            with torch.no_grad():
                grad[out_start:out_end,in_start:in_end] *= 0.
            return grad
        def bias_mask(grad):
            with torch.no_grad():
                grad[out_start:out_end] *= 0
            return grad
        self.hooks.append(self.weight.register_hook(weight_mask))
        if self.bias != None:
            self.hooks.append(self.bias.register_hook(bias_mask))

    def unset_mask(
        self
        ):
        for hook in self.hooks:
            hook.remove()

    def clear_rectange_weight(
        self,
        input_range_percentage: Tuple[float,float],
        output_range_percentage:Tuple[float,float],
        clear_bias=False
        ):
        in_start_per, in_end_per = input_range_percentage
        out_start_per, out_end_per = output_range_percentage
        in_start, out_start = (int)(in_start_per * self.in_features), (int)(out_start_per * self.out_features)
        in_end, out_end = (int)(in_end_per * self.in_features), (int)(out_end_per * self.out_features)
        with torch.no_grad():
            self.weight[out_start:out_end,in_start:in_end] *= 0
            if clear_bias and self.bias != None:
                self.bias[out_start:out_end] *= 0

    def uniform_Norm(self):
        with torch.no_grad():
            self.weight /= self.weight.norm(1,dim=1,keepdim=True)
