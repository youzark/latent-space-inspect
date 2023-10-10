#!/usr/bin/env python3
import torch
from torch import nn
import torch.optim as optim


BATCH=10
IN=5
OUT=5
HEIGHT=3
WIDTH=3

input = torch.randn(BATCH,IN,WIDTH,HEIGHT)


conv = nn.Conv2d(
    in_channels=IN,
    out_channels=OUT,
    kernel_size=3,
    padding=1
)


crit = nn.MSELoss()

optim = optim.Adam(conv.parameters(),lr=0.1)

output = conv(input)
loss = crit(output,input)

optim.zero_grad()
loss.backward()
print(conv.bias.grad.shape)


