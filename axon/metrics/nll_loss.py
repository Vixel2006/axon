import numpy as np
from axon.core.tensor import Tensor
from axon.functions import sum, mean, from_data

def nll_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    loss = -sum(predictions * targets, dim=1)
    return mean(loss)

