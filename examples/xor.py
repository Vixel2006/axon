from idrak.core.tensor import Tensor
import idrak.nn as nn
import idrak.metrics as metrics
import idrak.optim as optim
from idrak.data.dataset import Dataset
from idrak.functions import *

# Define the dataset
class XorSet(Dataset):
    def __init__(self):
        self.x = [Tensor((2,), [0, 0]), Tensor((2,), [0,1]), Tensor((2,), [1, 0]), Tensor((2,), [1,1])]
        self.y = [Tensor((1,), [0]), Tensor((1,), [1]), Tensor((1,), [1]), Tensor((1,), [0])]


    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

xorset = XorSet()

# Define the tanh function for the final leyar
def tanh(x: Tensor) -> Tensor:
    return x >> exp() >> sub(-x >> exp()) >> div(x >> exp() >> add(-x >> exp()))

# Here we define the model
model = nn.Sequential([
    nn.Linear(2, 4),
])

# Here we define the optimizer
optimizer = optim.sgd(model.params, 0.01)

# Here we do the loop
for i in range(10):
    for input, truth in xorset:
        optim.zero_grad()

        pred = model(input)

        pred.backward()

        optim.step()

print(model(Tensor((2,), [0, 0])))

