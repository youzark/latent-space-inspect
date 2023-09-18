#!/usr/bin/env python3
import torch
from einops import rearrange

x = torch.randn(2,3)
print(x)

x[:,[0,2]] = x[:,[2,0]] # swap column
print(x)

x = torch.randn(2,3)
print(x)
x[[0,1],:] = x[[1,0],:] # swap row
print(x)

tensor = torch.randn(10)

# Get the indices that would sort the tensor
sorted_indices = torch.argsort(tensor)

# For descending order, you can reverse the indices
descending_indices = sorted_indices.flip(dims=[0])

print("Tensor:", tensor)
print("Indices for sorted tensor:", sorted_indices)
print("Indices for descending order:", descending_indices)

tensor = torch.randn(3,3)
print(tensor)
tensor = rearrange(tensor, "a b -> b a")
print(tensor)
