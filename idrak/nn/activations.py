from idrak.nn.module import Module
from idrak.functions import exp
from idrak.core.tensor import Tensor

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return 1 / (1 + exp(-x))
