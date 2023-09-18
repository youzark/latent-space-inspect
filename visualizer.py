import torch

import matplotlib.pyplot as plt
from PIL import Image

import io
from abc import ABC, abstractmethod

from torch.utils.tensorboard.writer import SummaryWriter

from torchvision import transforms
from torchvision.utils import make_grid

from einops import rearrange

class PlotWriter(ABC):
    def __init__(
        self,
        tag:str,
        writer:SummaryWriter
        ):
        self.step= 0
        self.writer= writer
        self.tag= tag
        self.buf = io.BytesIO()

    @abstractmethod
    def plot(self, tensor:torch.Tensor):
        pass

class LinePloter(PlotWriter):
    def plot(self, tensor:torch.Tensor):
        tensor = tensor.to("cpu")
        plt.plot(tensor)
        plt.savefig(self.buf, format="jpeg")
        plt.clf()
        self.buf.seek(0)
        image = Image.open(self.buf)
        transformation = transforms.ToTensor()
        self.writer.add_image(
            tag= self.tag,
            img_tensor= transformation(image),
            global_step=self.step
        )
        self.buf.seek(0)
        self.buf.truncate(0)
        self.step += 1


class BarPloter(PlotWriter):
    def plot(self, tensor:torch.Tensor):
        tensor = tensor.to("cpu")
        plt.bar(range(len(tensor)),tensor)
        plt.savefig(self.buf, format="jpeg")
        plt.clf()
        self.buf.seek(0)
        image = Image.open(self.buf)
        transformation = transforms.ToTensor()
        self.writer.add_image(
            tag= self.tag,
            img_tensor= transformation(image),
            global_step=self.step
        )
        self.buf.seek(0)
        self.buf.truncate(0)
        self.step += 1

class HeatMapPloter(PlotWriter):
    def plot(self, tensor:torch.Tensor):
        tensor = tensor[:1000]
        tensor = rearrange(tensor, "n c -> c n")
        tensor = tensor.to("cpu")
        plt.figure(figsize=(480,6))
        plt.imshow(tensor, cmap="cool", aspect="auto")
        plt.savefig(self.buf, format="jpeg")
        plt.clf()
        plt.figure(figsize=(6.4,4.8))
        self.buf.seek(0)
        image = Image.open(self.buf)
        transformation = transforms.ToTensor()
        self.writer.add_image(
            tag= self.tag,
            img_tensor= transformation(image),
            global_step=self.step
        )
        self.buf.seek(0)
        self.buf.truncate(0)
        self.step += 1


class Layer1KernelVisualizer(PlotWriter):
    """
    inspect what pattern get captured
    """
    def plot(
        self,
        weight:torch.Tensor,
        ):
        weight = weight.to("cpu")

        img_grid = make_grid(weight,
                             nrow = 6,
                             normalize=True)
        self.writer.add_image(
            tag= self.tag,
            img_tensor= img_grid,
            global_step=self.step
        )
        self.step += 1


