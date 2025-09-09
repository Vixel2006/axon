from idrak.nn.module import Module
from idrak.core.tensor import Tensor
from idrak.ops.uop import UOp
from idrak.functions import relu, exp, sum
import ctypes

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return 1 / (1 + exp(-x))


class Softmax(Module):
    def forward(self, x: Tensor) -> Tensor:
        return exp(x) / sum(exp(x))

