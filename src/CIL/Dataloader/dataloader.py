from torch.utils.data import DataLoader,Subset
from torchvision import transforms, datasets

class CILMNIST:
    def __init__(
        self,
        num_of_task:int=5,
        root="./dataset/",
        ):
        transformation = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.ToTensor(),
                # transforms.Normalize(
                #     [0.5 for _ in range(IMG_CHANNELS)], [0.5 for _ in range(IMG_CHANNELS)]
                # )
            ]
        )

        train_dataset = datasets.MNIST(
            root=root,
            train= True,
            transform=transformation,
            download=True
        )

        test_dataset = datasets.MNIST(
            root=root,
            train=False,
            transform=transformation,
            download=True
        )

        classes_per_task = 10 // num_of_task

        task_train_indices = [[] for _ in range(num_of_task)]
        task_test_indices = [[] for _ in range(num_of_task)]
        for index,label in enumerate(train_dataset.targets):
            task_train_indices[label//classes_per_task].append(index)
        for index,label in enumerate(test_dataset.targets):
            task_test_indices[label//classes_per_task].append(index)

        self.task_train_subset = [Subset(train_dataset, train_indices) for train_indices in task_train_indices]
        self.task_test_subset = [Subset(test_dataset, test_indices) for test_indices in task_test_indices]
    
    def get_train_data_loader(
        self,
        batch_size,
        task_id,
        shuffle = True,
        ):
        return DataLoader(
            dataset=self.task_train_subset[task_id],
            batch_size= batch_size,
            shuffle= shuffle,
        )

    def get_test_data_loader(
        self,
        batch_size,
        task_id,
        shuffle = True,
        ):
        return DataLoader(
            dataset=self.task_test_subset[task_id],
            batch_size= batch_size,
            shuffle= shuffle,
        )

    def get_data_loader(
        self,
        batch_size,
        task_id,
        shuffle = True,
        ):
        return self.get_train_data_loader(batch_size,task_id,shuffle), self.get_test_data_loader(batch_size,task_id,shuffle)

