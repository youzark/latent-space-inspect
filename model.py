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
        self.in_channels = in_channels
        self.out_channels = out_channels
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
            weight[range(self.out_channels),:,:,:] = weight[index,:,:,:]

    def rearrange_input(self, index):
        with torch.no_grad():
            weight = self.conv.weight
            weight[:,range(self.in_channels),:,:] = weight[:,index,:,:]





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
        self.softmax = nn.Softmax()

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
            out_channels=24,
            kernel_size=7,
            stride=2,
        )
        self.layer2 = BasicBlock(
            in_channels=24,
            out_channels=48,
            kernel_size=3,
            stride=2,
        )
        self.layer3 = BasicBlock(
            in_channels=48,
            out_channels=48,
            kernel_size=3,
            stride=1,
        )
        self.layer4 = BasicBlock(
            in_channels=48,
            out_channels=96,
            kernel_size=3,
            stride=2,
        )
        self.layer5 = BasicBlock(
            in_channels=96,
            out_channels=96,
            kernel_size=3,
            stride=1,
        )
        self.layers = nn.Sequential(
            self.layer1,self.layer2,self.layer3,self.layer4,self.layer5
        )
        self.classifier = ClassificationHead(
            in_features=96*8*8,
            class_number=class_number
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        B, C, H, W = x5.shape
        x = rearrange(x5, "b c h w -> b (c h w)")
        prob = self.classifier(x)
        return prob,x1,x2,x3,x4,x5
