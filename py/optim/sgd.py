from .optimizer import Optimizer
from py.idrak_bindings.c_wrapper_functions import c_sgd, c_zero_grad
from py.core.tensor import Tensor

class SGD(Optimizer):
    def __init__(self, params: list[Tensor], lr: float):
        self.params = [param._c_tensor for param in params]
        self.num_params = len(params)
        self.lr = lr
    
    def step(self):
        c_sgd(self.params, self.num_params, self.lr)

    def zero_grad(self):
        c_zero_grad(self.params, self.num_params)

if __name__ == "__main__":
    a = Tensor((1,2), [2,3])
    b = Tensor((1,2), [3,4])

    c = a * b

    c.backward()
    
    print("Backward")
    print(a.grad)
    print(b.grad)
    print("Backward")
    optim = SGD([a, b], 0.1)

    optim.step()

    print(a)
    print(b)

    optim.zero_grad()

    print(a.grad)
    print(b.grad)
