#!/usr/bin/env python3
import torch
from torch import nn
import torch.optim as optim
from einops import rearrange
from torchvision.utils import make_grid
from model import CNN

from torchvision import datasets, transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from visualizer import LinePloter, BarPloter, HeatMapPloter,Layer1KernelVisualizer, ScalerPloter,PlotWriter

from structure_modification import rearrage_feature_dimension, truncate_feature_dimension

from latent_analyzer import Analyzer

from typing import Dict

DEVICE = "cuda"
LEARNING_RATE = 3e-4
BATCH_SIZE = 256
IMG_CHANNELS = 1
EPOCH = 100

transformation = transforms.Compose(
    [
        transforms.Resize(64),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     [0.5 for _ in range(IMG_CHANNELS)], [0.5 for _ in range(IMG_CHANNELS)]
        # )
    ]
)

writer = SummaryWriter()

group_name = "clipping_trival"
visiualizer : Dict[str,PlotWriter] = {
    "acc": ScalerPloter(
        tag = f"{group_name}/acc",
        writer=writer
    ),
    "feature ploter": BarPloter(
        tag = f"{group_name}/feature distribution",
        writer=writer
    ),
}

# layer1_BarPloter_ascending = BarPloter(
#     tag = "Feature Dist of layer 1 with ascending order",
#     writer= writer
# )

# layer1_BarPloter_ascending_trunc = BarPloter(
#     tag = "Feature Dist of layer 1 with ascending order after truncate",
#     writer= writer
# )

# layer1_HeatMapPloter = HeatMapPloter(
#     tag = "Feature dist of single inference",
#     writer= writer
# )

# layer1_kernelPloter = Layer1KernelVisualizer(
#     tag="leading pattern in layer1",
#     writer = writer
# )

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

analyzer = Analyzer(
    model= model,
    loader= test_loader
)

for epoch in range(EPOCH):
    model.train()
    for batch_idx,(image, labels) in enumerate(loader):
        image = image.to(DEVICE)
        labels = labels.to(DEVICE)
        prob_dist,latents = model(image)
        loss = criterion(prob_dist,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{EPOCH}]",
                f"Loss : {loss:.4F}",
            )

    analyzer.reevaluate()
    visiualizer["acc"].plot(analyzer.passed_case()/analyzer.total_case())



    # model.eval()
    # with torch.no_grad():
    #     in_c, out_c = model.layer1.get_shape()
    #     x1_col = torch.empty(0,out_c,32,32).to(DEVICE)
    #     correct=0
    #     for data, target in test_loader:
    #         data = data.to(DEVICE)
    #         target = target.to(DEVICE)
    #         prob_dist,x1,x2,x3,x4,x5 = model(data)
    #         x1_col = torch.cat((x1_col,x1),dim=0)

    #         pred = prob_dist.argmax(dim=1, keepdim=True)
    #         correct += pred.eq(target.view_as(pred)).sum().item()

    #     print(f"Acc Before Trunc: {correct}/{len(test_loader.dataset)}")

    #     x1_col = rearrange(x1_col, "b c h w -> (b h w) c")
    #     softmax = nn.Softmax(dim=1)
    #     x1_col = softmax(x1_col)
    #     x1_col_mean = torch.mean(x1_col,dim=0)

    #     ascending_indices = torch.argsort(x1_col_mean,descending=False)

        # rearrage_feature_dimension(
        #     model = model,
        #     layer_idx= 1,
        #     new_indices= ascending_indices
        # )
        # if epoch % 10 == 0:
        #     truncate_feature_dimension(
        #         model = model,
        #         layer_idx= 1,
        #         num_trunc=1,
        #     )

        # in_c, out_c = model.layer1.get_shape()
        # x1_col = torch.empty(0,out_c,32,32).to(DEVICE)
        # correct=0
        # for data, target in test_loader:
        #     data = data.to(DEVICE)
        #     target = target.to(DEVICE)
        #     prob_dist,x1,x2,x3,x4,x5 = model(data)
        #     x1_col = torch.cat((x1_col,x1),dim=0)

        #     pred = prob_dist.argmax(dim=1, keepdim=True)
        #     correct += pred.eq(target.view_as(pred)).sum().item()

        # x1_col = rearrange(x1_col, "b c h w -> (b h w) c")
        # x1_col = softmax(x1_col)
        # x1_col_mean = torch.mean(x1_col,dim=0)

        # layer1_BarPloter_ascending_trunc.plot(x1_col_mean)
        # print(f"Acc after trunc: {correct}/{len(test_loader.dataset)}")


        # layer1_kernelPloter.plot(model.layer1.conv.weight)



        # x1_col = torch.empty(0,16,32,32).to(DEVICE)
        # correct=0
        # for data, target in test_loader:
        #     data = data.to(DEVICE)
        #     target = target.to(DEVICE)
        #     prob_dist,x1,x2,x3,x4,x5 = model(data)
        #     x1_col = torch.cat((x1_col,x1),dim=0)

        #     pred = prob_dist.argmax(dim=1, keepdim=True)
        #     correct += pred.eq(target.view_as(pred)).sum().item()

        # print(f"Acc After Swap: {correct}/{len(test_loader.dataset)}")
#         x1_col = rearrange(x1_col, "b c h w -> (b h w) c")
#         softmax = nn.Softmax(dim=1)
#         x1_col = softmax(x1_col)
#         x1_col_mean = torch.mean(x1_col,dim=0)
#         layer1_LinePloter_after_swap.plot(x1_col_mean)
