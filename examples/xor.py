import sys
from idrak.core.tensor import Tensor
import idrak.nn as nn
import idrak.metrics as metrics
import idrak.optim as optim
from idrak.data.dataset import Dataset
from idrak.functions import *
from idrak.idrak_bindings.c_wrapper_functions import c_set_debug_mode
from idrak.nn.activations import Tanh, Sigmoid, ReLU

# Check for DEBUG argument
if "DEBUG=1" in sys.argv:
    c_set_debug_mode(1)
    sys.argv.remove("DEBUG=1")
else:
    c_set_debug_mode(0)

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


# Here we define the model
model = nn.Sequential(
    nn.Linear(2, 4, bias=False),
    Tanh(),
    nn.Linear(4, 1, bias=False),
    Sigmoid()
)

# Here we define the optimizer
optimizer = optim.SGD(model.params, 0.01)
print(model.params)
# Here we do the loop
for i in range(10):
    for input, truth in xorset:
        optimizer.zero_grad()

        pred = model(input)

        loss = metrics.bce(pred, truth)

        loss.backward()

        optimizer.step()

        print(model.params)

    truth = model(Tensor((1,2), [1, 0]))


    truth.realize()

    print(truth)
