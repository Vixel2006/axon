from .optimizer import Optimizer
from py.core.tensor import Tensor
from py.functions import zeros
from py.elnawah_bindings.c_wrapper_functions import c_adam, c_zero_grad

class Adam(Optimizer):
    def __init__(self, params: list[Tensor], lr: float, betas: tuple[float, float] = (0.9, 0.999), epsilon = 1e-8):
        self.params = [param._c_tensor for param in params]
        self.num_params = len(self.params)
        self._mt_tensors = [zeros(param.shape) for param in params]
        self._vt_tensors = [zeros(param.shape) for param in params]
        self.mt = [t._c_tensor for t in self._mt_tensors]
        self.vt = [t._c_tensor for t in self._vt_tensors]
        self.lr = lr
        self.betas = betas
        self.epsilon = epsilon
        self.time_step = 1

    def step(self):
        c_adam(self.params, self.mt, self.vt, self.num_params, self.time_step, self.lr, self.betas[0], self.betas[1], self.epsilon)
        self.time_step += 1

    def zero_grad(self):
        c_zero_grad(self.params, self.num_params)


if __name__ == "__main__":
    a = Tensor((1,2), [2,3], requires_grad=True)
    b = Tensor((1,2), [3,4], requires_grad=True)

    optim = Adam([a, b], 0.01) # Use a smaller learning rate for stability

    print("Initial a:", a)
    print("Initial b:", b)

    for i in range(100): # Run for 100 iterations
        # Recompute c in each iteration
        c = a * b
        
        # Compute gradients
        c.backward()
        
        # Update parameters
        optim.step()
        

        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}:")
            print("a:", a)
            print("b:", b)
            print("a.grad:", a.grad)
            print("b.grad:", b.grad)

        # Zero gradients for the next iteration
        optim.zero_grad()

