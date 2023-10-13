import torch
import torch.nn as nn
from typing import Dict,List,Tuple

class BatchNorm2d(nn.BatchNorm2d):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None
        ):
        super().__init__(
            num_features= num_features,
            eps= eps,
            momentum= momentum,
            affine= affine,
            track_running_stats= track_running_stats,
            device= device,
            dtype= dtype,
        )
        self.hooks = []
        

    def rearrange_output(self, index):
        with torch.no_grad():
            self.weight[:] = self.weight[index]
            if self.bias != None:
                self.bias[:] = self.bias[index]
            if self.running_var:
                self.running_var[:] = self.running_var[index]
            if self.running_mean:
                self.running_mean[:] = self.running_mean[index]

    def freeze_range_backward(
        self,
        output_range_percentage:Tuple[float,float]
        ):
        out_start_per, out_end_per = output_range_percentage
        out_start, out_end = (int)(out_start_per * self.num_features), (int)(out_end_per * self.num_features)
        def bias_mask(grad):
            with torch.no_grad():
                grad[out_start:out_end] *= 0
            return grad
        self.hooks.append(self.weight.register_hook(bias_mask))
        if self.bias != None:
            self.hooks.append(self.bias.register_hook(bias_mask))

    def unset_mask(
        self
        ):
        for hook in self.hooks:
            hook.remove()
