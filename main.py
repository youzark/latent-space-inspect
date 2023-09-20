#!/usr/bin/env python3
from Trainer import TrainingDynamic
from visualizer import ScalerPloter
from tensorboardX import SummaryWriter

DEVICE = "cuda"
LEARNING_RATE = 3e-4
BATCH_SIZE = 256
EPOCH = 100
LAYER = 4
writer = SummaryWriter(
    log_dir="./runs/trunc_test_2"
)

train_without_truncate = TrainingDynamic(device="cuda:7")

group_name = "truncate_trivial_half2"
acc_normal_1 = ScalerPloter(
    group=group_name,
    tag = f"normal train",
    writer= writer,
)
train_truncate_trivial = TrainingDynamic(
    device="cuda:6",
    stateDict=train_without_truncate.model.state_dict(),
)
acc_before_truncate_trivial = ScalerPloter(
    group=group_name,
    tag = f"before truncate trivial",
    writer = writer
)
acc_after_truncate_trivial = ScalerPloter(
    group=group_name,
    tag = f"after truncate trivial",
    writer = writer
)

group_name = "truncate_significant_half2"
acc_normal_2 = ScalerPloter(
    group=group_name,
    tag = f"normal train",
    writer= writer,
)
train_truncate_significant = TrainingDynamic(
    device="cuda:5",
    # stateDict=train_truncate_trivial.model.state_dict(),
    stateDict=train_without_truncate.model.state_dict(),
)
acc_before_clip_significant = ScalerPloter(
    group=group_name,
    tag = f"before truncate significant",
    writer = writer
)

acc_after_truncate_significant = ScalerPloter(
    group=group_name,
    tag = f"after truncate significant",
    writer = writer
)


for epoch in range(100):
    train_without_truncate.train()
    acc_normal_1.plot(train_without_truncate.acc())
    acc_normal_2.plot(train_without_truncate.acc())

    train_truncate_trivial.train()
    acc_before_truncate_trivial.plot(train_truncate_trivial.acc())
    print("epoch:",epoch)
    if epoch % 10 != 0:
        acc_after_truncate_trivial.plot(train_truncate_trivial.acc())
    else:
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
            print("Layer:",layer_idx,
                  "Feature dimension prev",train_truncate_trivial.model.layers[layer_idx-1].get_shape(),
                  "Feature dimension next",train_truncate_trivial.model.layers[layer_idx-1].get_shape(),
                 )
        acc_after_truncate_trivial.plot(train_truncate_trivial.acc())

    train_truncate_significant.train()
    acc_before_clip_significant.plot(train_truncate_significant.acc())
    if epoch % 10 != 0:
        acc_after_truncate_significant.plot(train_truncate_significant.acc())
    else:
        train_truncate_significant.SortDimension(
            layer=1,
            descending=False,
        )
        train_truncate_significant.TruncateDimension(
            layer=1
        )
        acc_after_truncate_significant.plot(train_truncate_significant.acc())
