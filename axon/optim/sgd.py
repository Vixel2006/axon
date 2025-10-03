from .optimizer import Optimizer
from axon.axon_bindings.c_wrapper_functions import c_sgd, c_zero_grad
from axon.core.tensor import Tensor
from axon.metrics import bce

class SGD(Optimizer):
    def __init__(self, params: list[Tensor], lr: float):
        self.params = [param for param in params]
        self.num_params = len(params)
        self.lr = lr
    
    def step(self):
        params_ptr = [param.c_tensor_ptr for param in self.params]
        c_sgd(params_ptr, self.num_params, self.lr)

    def zero_grad(self):
        params_ptr = [param.c_tensor_ptr for param in self.params]
        c_zero_grad(params_ptr, self.num_params)



