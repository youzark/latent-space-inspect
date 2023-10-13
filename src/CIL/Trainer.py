from torch import nn,optim
import torch.optim as optim
from Module import CILCNN

from dataloader import CILMNIST

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from visualizer import PlotWriter,ScalerPloter,BarPloter


from latent_analyzer import Analyzer
from typing import Dict, Any, List, Tuple
class Trainer:
    def __init__(
        self,
        model:CILCNN,
        loader:DataLoader,
        criterion:CrossEntropyLoss,
        optimizer:optim.Optimizer,
        task_id:int|None = None
        ):

        self.model = model
        self.loader = loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = next(model.parameters()).device
        self.epoch = 0
        self.task_id = task_id


    def __call__(self):
        if self.task_id == None:
            raise Exception("Please set task id before start training!")
        self.model.train()
        loss = None
        for batch_idx,(image, labels) in enumerate(self.loader):
            image = image.to(self.device)
            labels = labels.to(self.device)
            prob_dist = self.model(image)
            loss = self.criterion(prob_dist,labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.epoch += 1
        return self.epoch, loss


class TrainingDynamic:
    def __init__(
        self,
        batch_size:int = 256,
        learning_rate:float = 3e-4,
        device:str = "cuda",
        stateDict: None|Dict[str,Any] = None,
        num_of_task:int = 5,
        data_set_root="./dataset/"
        ):
        self.num_of_task = num_of_task
        self.batch_size = batch_size

        self.mnist = CILMNIST(
            num_of_task= self.num_of_task,
            root=data_set_root,
        )

        self.test_loaders = [ self.mnist.get_test_data_loader(batch_size,task_id,shuffle=False) for task_id in range(num_of_task) ]
        self.train_loaders = [ self.mnist.get_train_data_loader(batch_size,task_id) for task_id in range(num_of_task) ]

        self.model = CILCNN().to(device)

        if stateDict:
            self.model.load_state_dict(stateDict)

        optimizer = optim.SGD(
            params= self.model.parameters(),
            lr = learning_rate,
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


        self.trainers = [ Trainer(model=self.model,
                                 loader = self.train_loaders[task_id],
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 task_id=task_id) for task_id in range(self.num_of_task) ]

        self.analyzers = [ Analyzer(model=self.model,
                                    loader = self.test_loaders[task_id]) for task_id in range(self.num_of_task)]
        self.learned_task_ids = []

        self.ploters:List[Tuple[PlotWriter,Any]] = []


    def train(
        self,
        task_id,
        ):
        if task_id not in self.learned_task_ids:
            self.learned_task_ids.append(task_id)
        self.trainers[task_id]()

    def evaluate(
        self,
        task_id
        ):
        self.analyzers[task_id].evaluate()

    # def SortDimension(
    #     self,
    #     layer:int,
    #     descending:bool = True,
    #     eval_task_id:int|None = None
    #     ):
    #     if eval_task_id and eval_task_id in self.learned_task_ids:
    #         self.evaluate(task_id=eval_task_id)
    #     else:
    #         raise Exception("Learn task before Evaluate and Sort Dimension")
    #     sort_index = self.analyzers[eval_task_id].feature_sorted_index(
    #         layer= layer,
    #         descending= descending
    #     )
    #     self.model.rearrage_feature_dimension(
    #         layer_idx= layer,
    #         new_indices= sort_index,
    #     )
    #     self.evaluated = False

    # def TruncateDimension(
    #     self,
    #     layer:int,
    #     num_trunc:int=1,
    #     trunc_tail:bool=True
    #     ):
    #     self.evaluate()
    #     self.model.truncate_feature_dimension(
    #         layer_idx=layer,
    #         num_trunc=num_trunc,
    #         trunc_tail=trunc_tail
    #     )
    #     self.evaluated = False

    def acc(
        self,
        task_id,
        ) -> float:
        self.analyzers[task_id].evaluate()
        return self.analyzers[task_id].passed_case()/self.analyzers[task_id].total_case()

    def regist_acc_plotter(
        self,
        group:str, # line with same group appear in same chart
        tag:str, # separate lines in group
        ):
        self.ploters.append((ScalerPloter(
            tag = tag,
            group = group,
        ),self.acc))
        return self

    def regist_feature_plotter(
        self,
        ):
        pass

    def plot(
        self,
        eval_task_id
        ):
        for ploter, data_getter in self.ploters:
            ploter(data_getter(eval_task_id))

# def get_data(batch_size:int):
#     transformation = transforms.Compose(
#         [
#             transforms.Resize(64),
#             transforms.ToTensor(),
#             # transforms.Normalize(
#             #     [0.5 for _ in range(IMG_CHANNELS)], [0.5 for _ in range(IMG_CHANNELS)]
#             # )
#         ]
#     )

#     dataset = datasets.MNIST(
#         root="dataset/",
#         train= True,
#         transform=transformation,
#         download=True
#     )

#     train_loader = DataLoader(
#         dataset= dataset,
#         batch_size= batch_size,
#         shuffle= True
#     )

#     test_dataset = datasets.MNIST(
#         root="dataset/",
#         train=False,
#         transform=transformation,
#         download=True
#     )

#     test_loader = DataLoader(
#         dataset= test_dataset,
#         batch_size=1000
#     )

#     return train_loader, test_loader

