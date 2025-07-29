import numpy as np

from ..core import Tensor
from .module import Module


class BatchNorm2d(Module):
    def __init__(self, num_features: int, epsilon: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum

        self.gamma = Tensor(np.ones((1, num_features, 1, 1)), requires_grad=True)
        self.beta = Tensor(np.zeros((1, num_features, 1, 1)), requires_grad=True)

        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        if x.data.ndim != 4:
            raise ValueError(f"BatchNorm2d.forward expects 4D input, got {x.shape}")

        N, C, H, W = x.shape
        if C != self.num_features:
            raise ValueError(
                f"Channel mismatch: input has C={C}, but BatchNorm2d was created with num_features={self.num_features}"
            )

        if self.training:
            sum_x = x.sum(axis=(0, 2, 3), keepdims=True)
            batch_mean = sum_x / (N * H * W)

            var_x = ((x - batch_mean) ** 2).sum(axis=(0, 2, 3), keepdims=True)
            batch_var = var_x / (N * H * W)

            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * batch_mean.data
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * batch_var.data

            mean_to_use = batch_mean
            var_to_use = batch_var

        else:
            mean_to_use = Tensor(self.running_mean, requires_grad=False)
            var_to_use = Tensor(self.running_var, requires_grad=False)

        x_hat = (x - mean_to_use) / ((var_to_use + self.epsilon) ** 0.5)

        out = self.gamma * x_hat + self.beta
        return out
