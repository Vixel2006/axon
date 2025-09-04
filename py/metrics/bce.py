from py.functions import *
from py.core.tensor import Tensor

def bce(pred: Tensor, truth: Tensor, reduction: str = "mean") -> Tensor:
    out = -(truth * log(pred) + (1 - truth) * log(1 - pred))
    if reduction == "mean":
        return mean(out, 0)
    elif reduction == "sum":
        return sum(out, 0)

