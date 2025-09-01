from os import wait
from .module import Module
from py.core.tensor import Tensor
from py.functions import zeros, conv2d
from .init import *

class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...] | list[int],
        stride: tuple[int, int] | int = (1, 1),
        padding: int = 0,
        bias: bool = False
    ):
        self.kernel_size = (in_channels, out_channels, *kernel_size)
        self.weights = xavier_normal_(self.kernel_size, in_features=in_channels, out_features=out_channels)
        self.bias = None


        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding

        if bias:
            self.bias = zeros((out_channels))

    def forward(self, x: Tensor) -> Tensor:
        out = conv2d(x, self.weights, self.kernel_size, self.stride, self.padding)

        if self.bias:
            out += self.bias
        
        return out

if __name__ == "__main__":
    x = Tensor((1,1,4,4), [[[1,2,3,4], [1,2,3,4], [1,2,3,4], [1,2,3,4]]])
    layer = Conv2d(in_channels=1, out_channels=2, kernel_size=(2,2))

    pred = layer(x)

    pred.backward()

    print(layer.weights.grad)
