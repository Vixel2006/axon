import numpy as np
from axon.core.tensor import Tensor
from axon.functions import sum, mean, from_data

def nll_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    loss = -(predictions * targets)
    return mean(loss)

