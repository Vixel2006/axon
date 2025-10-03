from fajr.functions import *
from fajr.core.tensor import Tensor

def mse(pred: Tensor, truth: Tensor, reduction: str = "mean"):
    out = (pred - truth) ** 2
    return mean(out)

