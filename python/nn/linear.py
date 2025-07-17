import numpy as np

from ..core import (
    Tensor,
    init,
)
from .module import Module


class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int, has_bias: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.has_bias = has_bias

        self.W = Tensor(np.empty((in_dim, out_dim)), requires_grad=True)
        if self.has_bias:
            self.b = Tensor(np.empty(out_dim), requires_grad=True)
        else:
            self.b = None

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.W, a=np.sqrt(5))

        if self.b is not None:
            init.zeros_(self.b)

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.W
        if self.b is not None:
            out = out + self.b
        return out
