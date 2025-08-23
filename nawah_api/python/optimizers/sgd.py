from .optim import Optimizer
import cnawah as nw


class SGD(Optimizer):
    def __init__(self, params: list[nw.Tensor], lr: float):
        self.params = params
        self.lr = lr

    def step(self):
        nw.SGD(self.params, self.lr)

    def zero_grad(self):
        nw.zero_grad(self.params)
