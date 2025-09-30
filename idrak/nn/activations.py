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

    def reset_parameters(self):
        pass

class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return 1 / ((-x).exp() + 1)

    def reset_parameters(self):
        pass


