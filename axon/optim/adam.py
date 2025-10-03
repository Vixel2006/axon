from .optimizer import Optimizer
from axon.core.tensor import Tensor
from axon.functions import zeros
from axon.axon_bindings.c_wrapper_functions import c_adam, c_zero_grad

class Adam(Optimizer):
    def __init__(self, params: list[Tensor], lr: float, betas: tuple[float, float] = (0.9, 0.999), epsilon = 1e-8):
        self.params = [param.c_tensor_ptr for param in params]
        self.num_params = len(self.params)
        self._mt_tensors = [zeros(param.shape) for param in params]
        self._vt_tensors = [zeros(param.shape) for param in params]
        self.mt = [t.c_tensor_ptr for t in self._mt_tensors]
        self.vt = [t.c_tensor_ptr for t in self._vt_tensors]
        self.lr = lr
        self.betas = betas
        self.epsilon = epsilon
        self.time_step = 1

    def step(self):
        c_adam(self.params, self.mt, self.vt, self.num_params, self.time_step, self.lr, self.betas[0], self.betas[1], self.epsilon)
        self.time_step += 1

    def zero_grad(self):
        c_zero_grad(self.params, self.num_params)

