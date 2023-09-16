#!/usr/bin/env python3
import torch
from torch import nn
import torch.optim as optim
from einops import rearrange
from model import CNN

import matplotlib.pyplot as plt

from torchvision import datasets, transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from PIL import Image
import io

from visualizer import LinePloter, BarPloter, HeatMapPloter

DEVICE = "cuda"
LEARNING_RATE = 3e-4
BATCH_SIZE = 256
IMG_CHANNELS = 1
EPOCH = 100

transformation = transforms.Compose(
    [
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(IMG_CHANNELS)], [0.5 for _ in range(IMG_CHANNELS)]
        )
    ]
)

writer = SummaryWriter()
layer1_LinePloter = BarPloter(
    tag = "Feature Dist of layer 1",
    writer= writer
)
layer1_LinePloter_after_swap = BarPloter(
    tag = "Feature Dist of layer 1 after swap features",
    writer= writer
)
layer3_LinePloter = BarPloter(
    tag = "Feature Dist of layer 3",
    writer= writer
)

dataset = datasets.MNIST(
    root="dataset/",
    train= True,
    transform=transformation,
    download=True
)

loader = DataLoader(
    dataset= dataset,
    batch_size= BATCH_SIZE,
    shuffle= True
)

test_dataset = datasets.MNIST(
    root="dataset/",
    train=False,
    transform=transformation,
    download=True
)

test_loader = DataLoader(
    dataset= test_dataset,
    batch_size=1000
)

model = CNN(
    class_number= 10
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    params= model.parameters(),
    lr = LEARNING_RATE,
)

def plot_1d_bar(
    tensor:torch.Tensor,
    tag:str,
    step:int,
    writer:SummaryWriter,
    ):
    tensor = tensor.to("cpu")
    buf = io.BytesIO()
    plt.bar(range(len(tensor)),tensor)
    plt.savefig(buf, format="jpeg")
    plt.clf()
    buf.seek(0)
    image = Image.open(buf)
    transformation = transforms.ToTensor()
    writer.add_image(
        tag= tag,
        img_tensor= transformation(image),
        global_step=step
    )

for epoch in range(EPOCH):
    model.train()
    for batch_idx,(image, labels) in enumerate(loader):
        image = image.to(DEVICE)
        labels = labels.to(DEVICE)
        prob_dist,x1,x2,x3,x4,x5 = model(image)
        loss = criterion(prob_dist,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{EPOCH}]",
                f"Loss : {loss:.4F}",
            )

    model.eval()
    with torch.no_grad():
        x1_col = torch.empty(0,24,32,32).to(DEVICE)
        correct=0
        for data, target in test_loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            prob_dist,x1,x2,x3,x4,x5 = model(data)
            x1_col = torch.cat((x1_col,x1),dim=0)

            pred = prob_dist.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        print(f"Acc Before Swap: {correct}/{len(test_loader.dataset)}")

        x1_col = rearrange(x1_col, "b c h w -> (b h w) c")
        softmax = nn.Softmax(dim=1)
        x1_col = softmax(x1_col)
        x1_col_mean = torch.mean(x1_col,dim=0)
        layer1_LinePloter.plot(x1_col_mean)


        model.layer1.swap_output(0,21)
        model.layer1.swap_output(1,6)
        model.layer2.swap_input(0,21)
        model.layer2.swap_input(1,6)

        x1_col = torch.empty(0,24,32,32).to(DEVICE)
        correct=0
        for data, target in test_loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            prob_dist,x1,x2,x3,x4,x5 = model(data)
            x1_col = torch.cat((x1_col,x1),dim=0)

            pred = prob_dist.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        print(f"Acc After Swap: {correct}/{len(test_loader.dataset)}")

        x1_col = rearrange(x1_col, "b c h w -> (b h w) c")
        softmax = nn.Softmax(dim=1)
        x1_col = softmax(x1_col)
        x1_col_mean = torch.mean(x1_col,dim=0)
        layer1_LinePloter_after_swap.plot(x1_col_mean)


        model.layer1.swap_output(0,21)
        model.layer1.swap_output(1,6)
        model.layer2.swap_input(0,21)
        model.layer2.swap_input(1,6)
