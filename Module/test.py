#!/usr/bin/env python3
from logging import critical
import sys
import torch
from torch import nn
import torch.optim as optim
# from CILCNN import BasicConv

# print(sys.path)

# ten = torch.empty(0,3)
# sor = torch.randn(1,3)

# for i in range(10):
#     print(sor)
#     ten = torch.cat((ten,sor),dim=0)

# print(ten)
# criterion = nn.MSELoss()

# lin = nn.Linear(
#     in_features=5,
#     out_features=10,
#     bias= True
# )

# optimizer = optim.Adam(lin.parameters(),0.1)

# input = torch.randn(5,5)
# output = torch.randn(5,10)

# output_hat = lin(input)
# loss = criterion(output_hat,output)

# optimizer.zero_grad()
# loss.backward()


# bb = BasicConv(
#     in_channels= 10,
#     out_channels=20,
#     stride = 2,
#     kernel_size=3,
#     activation=nn.ReLU(),
# )

# input = torch.randn(5,10,3,3)
# output = torch.randn(5,20,2,2)

# optimizer = optim.Adam(bb.parameters(),lr=0.1)
# output_hat = bb(input)

# bb.freeze_old_unused_parameters((.2,.4),(.4,0.6))

# weight = bb.conv.weight.clone()

# loss = criterion(output,output_hat)
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()

# torch.set_printoptions(profile="full")
# print(weight[:,:,0,0] == bb.conv.weight[:,:,0,0])


lin = nn.Linear(in_features=10,out_features=20)
print(lin.weight.shape)

