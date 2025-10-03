from fajr.nn.module import Module
from fajr.core.tensor import Tensor
from fajr.ops.uop import UOp
from fajr.functions import log_softmax, sum, max, exp
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

class Softmax(Module):
    def forward(self, x: Tensor) -> Tensor:
        return softmax(x)
    
    def reset_parameters(self):
        pass

class LogSoftmax(Module):
    def forward(self, x: Tensor) -> Tensor:
        return log_softmax(x)

    def reset_parameters(self):
        pass


