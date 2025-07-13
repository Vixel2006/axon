import numpy as np
from neura import Tensor

from .optimizer import Optimizer


class Adam(Optimizer):
    def __init__(
        self,
        parameters: list[Tensor],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        epsilon: float = 1e-8,
    ):
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.epsilon = epsilon
        self.t = 0
        self.states = {}

        for param in self.parameters:
            self.states[id(param)] = {
                "m": np.zeros_like(param.data),
                "v": np.zeros_like(param.data),
            }

    def step(self):
        self.t += 1
        beta1, beta2 = self.betas

        for param in self.parameters:
            if param.grad is None:
                continue

            state = self.states[id(param)]
            grad = param.grad

            state["m"] = beta1 * state["m"] + (1 - beta1) * grad
            state["v"] = beta2 * state["v"] + (1 - beta2) * (grad**2)

            m_hat = state["m"] / (1 - beta1**self.t)
            v_hat = state["v"] / (1 - beta2**self.t)

            update = m_hat / (np.sqrt(v_hat) + self.epsilon)
            param.data -= self.lr * update

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()
