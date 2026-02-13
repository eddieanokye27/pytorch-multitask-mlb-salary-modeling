import torch
from assignment2 import multitask_training

model = multitask_training("multitask_data.csv")
print(type(model))
