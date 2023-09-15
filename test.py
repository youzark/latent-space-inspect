#!/usr/bin/env python3
import torch

x = torch.randn(3,3)
print(x)

x[:,[0,2]] = x[:,[2,0]] # swap column
print(x)

x = torch.randn(3,3)
x[[0,2]] = x[[2,0]] # swap row
print(x)
