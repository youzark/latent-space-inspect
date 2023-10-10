#!/usr/bin/env python3
import torch 
from Trainer import TrainingDynamic
TASK = 2
from dataloader import CILMNIST
import matplotlib.pyplot as plt

agent = TrainingDynamic(
    batch_size=256,
    num_of_task=TASK,
    learning_rate=0.05
)

agent.regist_acc_plotter(group="Class Incremental Learning",tag="CIL")

lin_weights = [agent.model.linear.weight.mean(dim=0)]
lin_weights_out = [agent.model.linear.weight.norm(dim=1)]

for task_id in range(2):
    task_range = [(0,0.7),(0.7,1),(0.7,0.8),(0.8,0.9),(0.9,1)]
    agent.model.freeze_backward(task_range[task_id])

    for epoch in range(3):
        agent.train(task_id)
        lin_weights.append(agent.model.linear.weight.mean(dim=0))
        lin_weights_out.append(agent.model.linear.weight.norm(dim=1))
        print(agent.model.linear.weight.norm(dim=1))
        agent.model.linear.uniform_Norm()
        # print(f"Epoch: {epoch} Agent:{task_id:} on task {task_id:} acc {agent.acc(task_id):.2f} ")
        for eval_task_id in range(task_id+1):

            es = "\t"*(eval_task_id+1)
            # print(f"Agent:{task_id:} eval on task {eval_task_id:}")
            # if eval_task_id == 0:
            #     agent.model.inspect = True
            #     agent.model.block1.inspect = True
            #     agent.model.block2.inspect = True
            #     agent.model.block3.inspect = True
            print(f"Epoch {epoch} Agent:{task_id:} on task {eval_task_id:} acc {es} {agent.acc(eval_task_id):.2f} ")
        # agent.model.inspect = False
        # agent.model.block1.inspect = False
        # agent.model.block2.inspect = False
        # agent.model.block3.inspect = False

    agent.model.unfreeze()

torch.set_printoptions(profile="full")
for pre, next in zip(lin_weights,lin_weights[1:]):
    print(next[:1792].mean())
    print(next[1792:].mean())
    # print(torch.all(pre[:1792]==next[:1792]))
    # print(torch.all(pre[1792:]==next[1792:]))
    print()

for pre, next in zip(lin_weights_out,lin_weights_out[1:]):
    print(next)
    print(pre == next)
    print()


# print(agent.model.latents)
# print(agent.model.block1.latents)
# print(agent.model.block2.latents)
# print(agent.model.block3.latents)

# print(torch.all(agent.model.tensors == agent.model.tensors[0],dim=0))
