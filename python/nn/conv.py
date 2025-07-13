from typing import Union

import numpy as np

from ..core import Tensor
from .module import Module


def _to_tuple(value: Union[int, tuple], n: int) -> tuple:
    """Converts an int or a tuple into a tuple of length n."""
    if isinstance(value, int):
        return (value,) * n
    elif isinstance(value, tuple) and len(value) == n:
        return value
    raise ValueError(f"Value must be an int or a tuple of length {n}, but got {value}")


from ..core import init  # Import our new init module


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        has_bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _to_tuple(kernel_size, 2)
        self.stride = _to_tuple(stride, 2)
        self.padding = _to_tuple(padding, 2)
        self.has_bias = has_bias

        weight_shape = (
            out_channels,
            in_channels,
            self.kernel_size[0],
            self.kernel_size[1],
        )

        self.W = Tensor(np.empty(weight_shape), requires_grad=True)
        if self.has_bias:
            self.b = Tensor(np.empty((out_channels,)), requires_grad=True)
        else:
            self.b = None

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.W, a=np.sqrt(5))
        if self.b is not None:
            init.zeros_(self.b)

    def forward(self, x: Tensor) -> Tensor:
        out = x.conv2d(self.W, padding=self.padding, stride=self.stride)

        if self.b is not None:
            bias_reshaped = self.b.view(1, self.out_channels, 1, 1)
            out = out + bias_reshaped
        return out


class Conv2dTranspose(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: Union[tuple, int] = 0,
        stride: Union[tuple, int] = 1,
        has_bias: bool = False,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.has_bias = has_bias

        weight_shape = (
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        scale = np.sqrt(2.0 / (self.in_channels * self.kernel_size * self.kernel_size))
        self.W = Tensor(
            np.random.randn(*weight_shape).astype(np.float32) * scale,
            requires_grad=True,
        )

        self.b = (
            Tensor(np.zeros((self.out_channels, 1, 1)), requires_grad=True)
            if self.has_bias
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        _, C_in, _, _ = x.shape

        if C_in != self.in_channels:
            raise ValueError(
                f"Input tensor channels {C_in} ≠ expected {self.in_channels}"
            )

        out = x.conv2dTranspose(self.W, padding=self.padding, stride=self.stride)

        if self.has_bias:
            out += self.b

        return out
