from idrak.nn.module import Module
from idrak.core.tensor import Tensor
from idrak.ops.uop import UOp
from idrak.functions import *
import ctypes

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

    def reset_parameters(self):
        pass

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())

class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return 1 / ((-x).exp() + 1)

    def reset_parameters(self):
        pass


class Softmax(Module):
    def forward(self, x: Tensor) -> Tensor:
        x_max = max(x, dim=-1)
        exp_x = (x - x_max).exp()
        return exp_x / sum(exp_x, dim=-1)
