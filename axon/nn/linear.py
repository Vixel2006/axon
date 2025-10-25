from .module import Module
from axon.functions import *
from axon.core.tensor import Tensor
from axon.nn.init import xavier_uniform_, xavier_normal_


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.W = xavier_normal_((out_features, in_features), in_features, out_features)

        self.bias = bias

        if self.bias:
            self.B = zeros((1, out_features))
        else:
            self.B = None

    def forward(self, x):
        out = x @ self.W.T

        if self.bias:
            out += self.B

        return out
