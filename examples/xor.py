import sys
from idrak.core.tensor import Tensor
import idrak.nn as nn
import idrak.metrics as metrics
import idrak.optim as optim
from idrak.data.dataset import Dataset
from idrak.functions import *
from idrak.nn.activations import Tanh, Sigmoid, ReLU
from idrak.functions import *


# Define the dataset
class XorSet(Dataset):
    def __init__(self):
        self.x = [from_data((1, 2), [[0, 0]]), from_data((1, 2), [[0,1]]), from_data((1, 2), [[1, 0]]), from_data((1, 2), [[1,1]])]
        self.y = [from_data((1,1), [[0]]), from_data((1,1), [[1]]), from_data((1,1), [[1]]), from_data((1,1), [[0]])]


    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

xorset = XorSet()

# Here we define the model
model = nn.Sequential(
    nn.Linear(2, 2, bias=False),
    Tanh(),
    nn.Linear(2, 4, bias=False),
    Tanh(),
    nn.Linear(4, 1, bias=False),
    Sigmoid()
)

# Here we define the optimizer
optimizer = optim.SGD(model.params, 0.1)
# Here we do the loop
for i in range(1000):
    for inp, truth in xorset:
        optimizer.zero_grad()

        pred = model(inp)

        loss = metrics.bce(pred, truth)

        loss.backward()

        optimizer.step()

    if (i+1) % 10 == 0:
        print(f"Epoch {i+1} -> Loss: {loss}")

for inp, _ in xorset:
    pred = model(inp)

    pred.realize()

    if pred.data[0, 0] > 0.5:
        num = 1
    else:
        num = 0

    print(num)

