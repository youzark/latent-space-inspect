import torch

import matplotlib.pyplot as plt
from PIL import Image

import io
from abc import ABC, abstractmethod

from torch.utils.tensorboard.writer import SummaryWriter

from torchvision import transforms

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
        tensor = tensor.to("cpu")
        plt.bar(range(len(tensor)),tensor)
        plt.imshow(tensor, cmap="hot")
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
