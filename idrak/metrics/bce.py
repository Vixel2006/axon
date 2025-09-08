from idrak.functions import *
from idrak.core.tensor import Tensor

def bce(pred: Tensor, truth: Tensor, reduction: str = "mean") -> Tensor:
    out = -(truth * log(pred)) - ((1 - truth) * log(1 - pred))
    return out

