#!/usr/bin/env python3
from einops import rearrange
import torch
from torch import nn, tensor
import torch.optim as optim

import torchvision
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

import matplotlib.pyplot as plt
from PIL import Image
import io

from model import CNN

from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

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


step = 0
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
    correct=0
    with torch.no_grad():
        x1_col = torch.empty(0,24,32,32).to(DEVICE)
        for data, target in test_loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            prob_dist,x1,x2,x3,x4,x5 = model(data)
            x1_col = torch.cat((x1_col,x1),dim=0)

            pred = prob_dist.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    x1_col = rearrange(x1_col, "b c h w -> (b h w) c")
    softmax = nn.Softmax(dim=1)
    x1_col = softmax(x1_col)
    x1_col_mean = torch.mean(x1_col,dim=0)
    print(f"Acc Before Swap: {correct}/{len(test_loader.dataset)}")
    plot_1d_bar(
        tensor=x1_col_mean,
        tag="layer1 feature average map",
        step=step,
        writer=writer
    )
    print("finish!")



    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            prob_dist,x1,x2,x3,x4,x5 = model(data)
            pred = prob_dist.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(f"Acc After Swap: {correct}/{len(test_loader.dataset)}")
