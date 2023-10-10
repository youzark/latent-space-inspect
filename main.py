#!/usr/bin/env python3
from Trainer import TrainingDynamic
from tensorboardX import SummaryWriter
import json

with open("config.json","r") as f:
    config = json.load(
        fp = f,
    )

DEVICE = "cuda"
LEARNING_RATE = 3e-4
BATCH_SIZE = 256
EPOCH = 100
LAYER = 4

writer = SummaryWriter(
    log_dir="./runs/pruning_test"
)

group_all_comp = "train_loop"

train_without_truncate = TrainingDynamic(
    batch_size=BATCH_SIZE,
    device=DEVICE
).regist_acc_plotter(
    group_all_comp,
    "no truncation"
)

train_truncate_trivial = TrainingDynamic(
    batch_size=BATCH_SIZE,
    device=DEVICE,
    stateDict=train_without_truncate.model.state_dict()
).regist_acc_plotter(
    group= group_all_comp,
    tag = "Truncate Trivial Dimension"
)

train_truncate_significant = TrainingDynamic(
    batch_size=BATCH_SIZE,
    device=DEVICE,
    stateDict=train_without_truncate.model.state_dict(),
).regist_acc_plotter(
    group=group_all_comp,
    tag="Truncate Significant Dimension"
)

for epoch in range(100):
    train_without_truncate.train()
    train_truncate_trivial.train()
    train_truncate_significant.train()
    print("epoch:",epoch)
    if epoch % 10 == 0:
        for layer_idx in range(1,LAYER+1):
            train_truncate_trivial.SortDimension(
                layer=layer_idx,
            )
            layer_dim = train_truncate_trivial.model.layers[layer_idx].get_shape()[0]
            num_trunc = int(layer_dim * 0.1)
            train_truncate_trivial.TruncateDimension(
                layer=layer_idx,
                num_trunc=num_trunc
            )

        for layer_idx in range(1,LAYER+1):
            train_truncate_significant.SortDimension(
                layer=layer_idx,
                descending=False,
            )
            layer_dim = train_truncate_significant.model.layers[layer_idx].get_shape()[0]
            num_trunc = int(layer_dim * 0.1)
            train_truncate_significant.TruncateDimension(
                layer=layer_idx,
                num_trunc=num_trunc
            )

    train_without_truncate.plot()
    train_truncate_trivial.plot()
    train_truncate_significant.plot()
