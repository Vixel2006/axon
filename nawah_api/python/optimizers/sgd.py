from ..core import Tensor

def sgd(params: list[Tensor], lr: int, batch_size: int):
    for param in params
    param.data -= lr * param.grad

