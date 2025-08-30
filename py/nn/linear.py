from .module import Module
from py.core.tensor import Tensor

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.W = Tensor((out_features, in_features))

        self.bias = bias
        
        if self.bias:
            self.B = Tensor((1, out_features))

    def forward(self, x):
        out = x @ self.W.view([2, 3])

        if self.bias:
            out += self.B
        
        return out

if __name__ == "__main__":
    net = Linear(2, 3)

    x = Tensor((1, 2), [3,4])

    y = net.forward(x)

    y.realize()

    print(y)