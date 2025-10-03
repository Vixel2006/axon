import numpy as np
from fajr.core.tensor import Tensor
from fajr.functions import one_hot, sum, mean, from_data

def nll_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    loss = -(predictions * targets)
    return mean(loss)

