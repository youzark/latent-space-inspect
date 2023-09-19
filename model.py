import torch
from torch import nn
from einops import rearrange

class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels:int,
        out_channels:int,
        kernel_size:int = 3,
        stride:int=1,
        activation: nn.ReLU|None = nn.ReLU()
        ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            stride=stride,
            bias=False
        )
        # self.norm = nn.BatchNorm2d(
        #     num_features=out_channels
        # )
        self.activation = activation


    def forward(self,x):
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        return x

    def get_shape(self):
        return self.conv.in_channels, self.conv.out_channels

    def swap_output(self, i ,j):
        with torch.no_grad():
            weight = self.conv.weight
            weight[[i,j],:,:,:] = weight[[j,i],:,:,:]

    def swap_input(self, i ,j):
        with torch.no_grad():
            weight = self.conv.weight
            weight[:,[i,j],:,:] = weight[:,[j,i],:,:]

    def rearrange_output(self, index):
        with torch.no_grad():
            weight = self.conv.weight
            weight[range(self.conv.out_channels),:,:,:] = weight[index,:,:,:]

    def rearrange_input(self, index):
        with torch.no_grad():
            weight = self.conv.weight
            weight[:,range(self.conv.in_channels),:,:] = weight[:,index,:,:]

    def truncate_output(
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

            truncated_old_weight = self.conv.weight[:-num_trunc,:,:,:] if trunc_tail else self.conv.weight[num_trunc:,:,:,:]

            truncated_old_bias=None
            if self.conv.bias != None:
                truncated_old_bias = self.conv.bias[:-num_trunc] if trunc_tail else self.conv.bias[num_trunc:]

            self.conv = nn.Conv2d(
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



class ClassificationHead(nn.Module):
    def __init__(
        self,
        in_features:int,
        class_number:int
        ):
        super().__init__()
        self.linear = nn.Linear(
            in_features=in_features,
            out_features= class_number,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.linear(x)
        probDist = self.softmax(logits)
        return probDist

class CNN(nn.Module):
    """
    """
    def __init__(
        self,
        class_number:int
         ):
        super().__init__()
        self.layer1 = BasicBlock(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=2,
        )
        self.layer2 = BasicBlock(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=2,
        )
        self.layer3 = BasicBlock(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
        )
        self.layer4 = BasicBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
        )
        self.layer5 = BasicBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
        )
        self.layers = [
            self.layer1,self.layer2,self.layer3,self.layer4,self.layer5
        ]
        self.classifier = ClassificationHead(
            in_features=64*8*8,
            class_number=class_number
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        latents = (x1,x2,x3,x4,x5)
        x = rearrange(x5, "b c h w -> b (c h w)")
        prob = self.classifier(x)
        return prob, latents

    def rearrage_feature_dimension(
            self,
            layer_idx:int,
            new_indices:torch.Tensor
        ):
        self.layers[layer_idx-1].rearrange_output(new_indices)
        self.layers[layer_idx].rearrange_input(new_indices)

    def truncate_feature_dimension(
        self,
        layer_idx:int,
        num_trunc:int=1
        ):
        self.layers[layer_idx-1].truncate_output(num_trunc)
        self.layers[layer_idx].truncate_input(num_trunc)
