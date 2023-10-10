#!/usr/bin/env python3
from dataloader import CILMNIST
# i = 10
# n = 10
# print([n*i for i in range(n) if i < 10])

dataclass = CILMNIST(
    num_of_task=5
)

train_loader, test_loader = dataclass.get_data_loader(
    batch_size=32,
    task_id=1,
)

for _,(input,label) in enumerate(train_loader):
    print(label)
