from py.functions import *
from py.core.tensor import Tensor

def mse(pred: Tensor, truth: Tensor, reduction: str = "mean"):
    if reduction == "mean":
        return mean(((pred - truth) ** 2), dim=-1, keepdim=True)
    elif reduction == "sum":
        return sum(((pred - truth) ** 2), dim=-1, keepdim=True)

