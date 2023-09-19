import torch
from torch import nn
from model import CNN
from torch.utils.data import DataLoader

from einops import rearrange

class Analyzer:
    def __init__(
        self,
        model: CNN,
        loader: DataLoader,
        ):
        self.model = model
        self.loader = loader
        self.device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            self.correct=0
            for data, target in self.loader:
                data = data.to(self.device)
                target = target.to(self.device)
                prob_dist,self.latents = self.model(data)

                pred = prob_dist.argmax(dim=1, keepdim=True)
                self.correct += pred.eq(target.view_as(pred)).sum().item()

    def reevaluate(self):
        self.model.eval()
        self.device = next(self.model.parameters()).device
        with torch.no_grad():
            self.correct=0
            for data, target in self.loader:
                data = data.to(self.device)
                target = target.to(self.device)
                prob_dist,self.latents = self.model(data)

                pred = prob_dist.argmax(dim=1, keepdim=True)
                self.correct += pred.eq(target.view_as(pred)).sum().item()

    def passed_case(self):
        return self.correct

    def total_case(self):
        return len(self.loader.dataset)

    def mean_feature(
        self,
        layer:int,
        ):
        feature = rearrange(self.latents[layer], "b c h w -> (b h w) c")
        softmax = nn.Softmax(dim=1)
        feature = softmax(feature)
        return torch.mean(feature,dim=0)

    def feature_sorted_index(
        self,
        layer:int,
        descending:bool = True
        ):
        return torch.argsort(
            self.mean_feature(layer),
            descending=descending
        )


