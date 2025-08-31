from .module import Module
from py.functions import *
from py.core.tensor import Tensor
from py.nn.init import xavier_uniform_, xavier_normal_


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.W = xavier_normal_((out_features, in_features), in_features, out_features)

        self.bias = bias

        if self.bias:
            self.B = Tensor((1, out_features))

    def forward(self, x):
        out = x @ self.W.transpose(n=0, m=1)

        if self.bias:
            out += self.B

        return out


if __name__ == "__main__":
    net = Linear(2, 3, bias=True)

    x = Tensor((4, 2), [[3, 4], [3, 4], [4, 5], [6, 7]])

    y = net.forward(x)

    y.realize()
    print(y)
