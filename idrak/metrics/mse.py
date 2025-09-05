from idrak.functions import *
from idrak.core.tensor import Tensor

def mse(pred: Tensor, truth: Tensor, reduction: str = "mean"):
    if reduction == "mean":
        return mean((pred - truth) ** 2)
    elif reduction == "sum":
        return sum((pred - truth) ** 2)

