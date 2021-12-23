import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim

LEARNING_RATE = 0.001

def z_func(x, y):
    return np.sin(5 * np.cos(x) / 2) + np.cos(y)


x = np.arange(start=0, stop=1, step=0.0001, dtype=float)
y = np.arange(start=0, stop=1, step=0.0001, dtype=float)

input_array = torch.tensor(np.vstack((x, y)).T)
target = torch.tensor(z_func(x, y).T)



# Create a model
model = nn.Sequential(nn.Linear(2, 10),
   nn.ReLU(),
   nn.Linear(10, 1),
   nn.ReLU())

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
criterion = nn.NLLLoss()

for epoch in range(100):
    y_pred = model(input_array.float())
    loss = criterion(y_pred, target.long())
    print('epoch: ', epoch, ' loss: ', loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


y_pred = model(input_array.float())
print(y_pred)
print(target)

