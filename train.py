#!/usr/bin/env python3
import torch
from torch import nn
import torch.optim as optim

import torchvision
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from model import CNN

from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

DEVICE = "cuda"
LEARNING_RATE = 3e-4
BATCH_SIZE = 256
IMG_CHANNELS = 1
EPOCH = 100

transforms = transforms.Compose(
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
    transform=transforms,
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
    transform=transforms,
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

step = 0
for epoch in range(EPOCH):
    model.train()
    for batch_idx,(image, labels) in enumerate(loader):
        image = image.to(DEVICE)
        labels = labels.to(DEVICE)
        prob_dist = model(image)
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
        for data, target in test_loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            prob_dist = model(data)
            pred = prob_dist.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(f"Acc: {correct}/{len(test_loader.dataset)}")







