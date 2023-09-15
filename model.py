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
        self.norm = nn.BatchNorm2d(
            num_features=out_channels
        )
        self.activation = activation

    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

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
        layer1 = BasicBlock(
            in_channels=1,
            out_channels=24,
            kernel_size=7,
            stride=2,
        )
        layer2 = BasicBlock(
            in_channels=24,
            out_channels=48,
            kernel_size=3,
            stride=2,
        )
        layer3 = BasicBlock(
            in_channels=48,
            out_channels=48,
            kernel_size=3,
            stride=1,
        )
        layer4 = BasicBlock(
            in_channels=48,
            out_channels=96,
            kernel_size=3,
            stride=2,
        )
        layer5 = BasicBlock(
            in_channels=96,
            out_channels=96,
            kernel_size=3,
            stride=1,
        )
        self.layers = nn.Sequential(
            layer1,layer2,layer3,layer4,layer5
        )
        self.classifier = ClassificationHead(
            in_features=96*8*8,
            class_number=class_number
        )

    def forward(self, x):
        x = self.layers(x)
        B, C, H, W = x.shape
        x = rearrange(x, "b c h w -> b (c h w)")
        prob = self.classifier(x)
        return prob
