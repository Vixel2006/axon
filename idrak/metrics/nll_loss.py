from idrak.core.tensor import Tensor
from idrak.functions import *

def nll_loss(pred: Tensor, truth: Tensor, reduction="mean"):
    loss = -1 * pred * truth
    loss = loss.sum(dim=1)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss

