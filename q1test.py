import torch
from assignment2 import MultitaskNetwork

model = MultitaskNetwork()
x = torch.randn(4, 3)

y1, y2 = model(x)

print(y1.shape, y2.shape)
print(y1.sum(dim=1), y2.sum(dim=1))
