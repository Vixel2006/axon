from idrak.core.tensor import Tensor
import idrak.nn as nn
import idrak.metrics as metrics
import idrak.optim as optim
from idrak.data.dataset import Dataset
from idrak.functions import *

# Define the dataset
class XorSet(Dataset):
    def __init__(self):
        self.x = [Tensor((1, 2), [0, 0]), Tensor((1, 2), [0,1]), Tensor((1, 2), [1, 0]), Tensor((1, 2), [1,1])]
        self.y = [Tensor((1,), [0]), Tensor((1,), [1]), Tensor((1,), [1]), Tensor((1,), [0])]


    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

xorset = XorSet()

# Define the tanh function for the final leyar
def tanh(x: Tensor) -> Tensor:
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

# Here we define the model
model = nn.Sequential(
    nn.Linear(2, 4, bias=False),
    nn.Linear(4, 1, bias=False),
)

# Here we define the optimizer
optimizer = optim.SGD(model.params, 0.01)

# Here we do the loop
for i in range(10):
    for input, truth in xorset:
        optimizer.zero_grad()

        pred = model(input)


        pred.backward()

        optimizer.step()

truth = model(Tensor((1,2), [1, 0]))


truth.realize()

print(truth)
