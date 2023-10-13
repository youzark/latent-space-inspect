import torch
from Module import CILCNN
from torch.utils.data import DataLoader
from einops import rearrange

class Analyzer:
    def __init__(
        self,
        model: CILCNN,
        loader: DataLoader,
        ):
        self.model = model
        self.device = next(self.model.parameters()).device
        self.model.eval()
        self.loader = loader

    def evaluate(
        self,
        ):
        self.model.eval()

        def count_freq(tensor):
            counts = torch.bincount(tensor)
            numbers = torch.unique(tensor)
            return [ (n.item(), counts[n].item()) for n in numbers]

        with torch.no_grad():
            self.correct=0
            self.preds = []
            self.targets =  []
            self.dists = []
            for _,(data, target) in enumerate(self.loader):
                data = data.to(self.device)
                target = target.to(self.device)
                prob_dist = self.model(data)
                self.dists.append(prob_dist)
                

                pred = prob_dist.argmax(dim=1, keepdim=True)
                self.preds.append(pred)
                self.targets.append(target)
                # print(pred.view(-1)[:10])
                # print(target.view(-1)[:10])
                # print(target.view(-1)[:10] == pred.view(-1)[:10])
                self.correct += pred.eq(target.view_as(pred)).sum().item()
            print(count_freq(torch.cat(self.preds).view(-1)))
            print(count_freq(torch.cat(self.targets).view(-1)))

    def passed_case(self):
        return self.correct

    def total_case(
        self,
        ):
        return len(self.loader.dataset)

    # def mean_feature(
    #     self,
    #     layer:int,
    #     ):
    #     feature = rearrange(self.latents[layer-1], "b c h w -> (b h w) c")
    #     softmax = nn.Softmax(dim=1)
    #     feature = softmax(feature)
    #     return torch.mean(feature,dim=0)

    # def feature_sorted_index(
    #     self,
    #     layer:int,
    #     descending:bool = True
    #     ):
    #     return torch.argsort(
    #         self.mean_feature(layer),
    #         descending=descending
    #     )


