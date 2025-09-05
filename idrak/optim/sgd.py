from .optimizer import Optimizer
from idrak.idrak_bindings.c_wrapper_functions import c_sgd, c_zero_grad
from idrak.core.tensor import Tensor

class SGD(Optimizer):
    def __init__(self, params: list[Tensor], lr: float):
        self.params = [param._c_tensor for param in params]
        self.num_params = len(params)
        self.lr = lr
    
    def step(self):
        c_sgd(self.params, self.num_params, self.lr)

    def zero_grad(self):
        c_zero_grad(self.params, self.num_params)

