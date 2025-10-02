import numpy as np
from idrak.core.tensor import Tensor
from idrak.functions import one_hot, sum, mean, from_data

def nll_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    loss = -(predictions * targets)
    return mean(loss)

if __name__ == "__main__":
    pred = from_data((1, 5), [[0.20, 0.68, 0.1, 0.20, 0.10]])
    truth = from_data((1,5), [[0, 1, 0, 0, 0]])

    loss = nll_loss(pred, truth)

    loss.backward()

