from torch import nn,optim
import torch.optim as optim
from model import CNN

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchvision import transforms, datasets

from visualizer import PlotWriter,ScalerPloter,BarPloter

from latent_analyzer import Analyzer
from typing import Dict, Any, List, Tuple
class Trainer:
    def __init__(
        self,
        model:CNN,
        loader:DataLoader,
        criterion:CrossEntropyLoss,
        optimizer:optim.Adam
        ):

        self.model = model
        self.loader = loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = next(model.parameters()).device
        self.epoch = 0

    def __call__(self):
        self.model.train()
        loss = None
        for batch_idx,(image, labels) in enumerate(self.loader):
            image = image.to(self.device)
            labels = labels.to(self.device)
            prob_dist,_ = self.model(image)
            loss = self.criterion(prob_dist,labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.epoch += 1
        return self.epoch, loss

def get_data(batch_size:int):
    transformation = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     [0.5 for _ in range(IMG_CHANNELS)], [0.5 for _ in range(IMG_CHANNELS)]
            # )
        ]
    )
    dataset = datasets.MNIST(
        root="dataset/",
        train= True,
        transform=transformation,
        download=True
    )

    train_loader = DataLoader(
        dataset= dataset,
        batch_size= batch_size,
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

    return train_loader, test_loader


class TrainingDynamic:
    def __init__(
        self,
        batch_size:int = 256,
        learning_rate:float = 3e-4,
        device:str = "cuda",
        stateDict: None|Dict[str,Any] = None
        ):

        self.train_loader,self.test_loader = get_data(batch_size)

        self.model = CNN(
            class_number= 10
        ).to(device)

        if stateDict:
            self.model.load_state_dict(stateDict)

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            params= self.model.parameters(),
            lr = learning_rate,
        )

        self.analyzer = Analyzer(
            model= self.model,
            loader= self.test_loader
        )

        self.trainer = Trainer(
            model = self.model,
            loader = self.train_loader,
            criterion= self.criterion,
            optimizer= self.optimizer
        )

        self.ploters:List[Tuple[PlotWriter,Any]] = []

        self.evaluated = False

    def train(self):
        self.trainer()
        self.evaluated = False

    def evaluate(self):
        if not self.evaluated:
            self.analyzer.reevaluate()
        self.evaluated = True

    def SortDimension(
        self,
        layer:int,
        descending:bool = True
        ):
        self.evaluate()
        sort_index = self.analyzer.feature_sorted_index(
            layer= layer,
            descending= descending
        )
        self.model.rearrage_feature_dimension(
            layer_idx= layer,
            new_indices= sort_index,
        )
        self.evaluated = False

    def TruncateDimension(
        self,
        layer:int,
        num_trunc:int=1,
        trunc_tail:bool=True
        ):
        self.evaluate()
        self.model.truncate_feature_dimension(
            layer_idx=layer,
            num_trunc=num_trunc,
            trunc_tail=trunc_tail
        )
        self.evaluated = False

    def acc(
        self,
        ) -> float:
        self.evaluate()
        return self.analyzer.passed_case()/self.analyzer.total_case()

    def regist_acc_plotter(
        self,
        tag,
        group,
        ):
        self.ploters.append((ScalerPloter(
            tag = tag,
            group = group,
        ),self.acc))
        return self

    def plot(self):
        for ploter, data_getter in self.ploters:
            ploter(data_getter())

