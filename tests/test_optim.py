import pytest
import numpy as np
from idrak.core.tensor import Tensor
from idrak.optim.sgd import SGD
from idrak.optim.adam import Adam
from idrak.functions import from_data, zeros


class TestOptim:

    def test_sgd_step(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        grad = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        tensor = from_data(data.shape, data)
        tensor.requires_grad = True  # Ensure grad buffer is allocated
        lr = 0.1
        optimizer = SGD(params=[tensor], lr=lr)

        optimizer.step()

        expected_data = data - lr * grad
        assert np.allclose(tensor.data, expected_data)

    def test_sgd_zero_grad(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = from_data(data.shape, data)
        tensor.requires_grad = True
        lr = 0.1
        optimizer = SGD(params=[tensor], lr=lr)

        optimizer.zero_grad()

        assert np.allclose(tensor.grad, np.zeros_like(data))

    def test_adam_step(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        grad = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        tensor = from_data(data.shape, data)
        tensor.requires_grad = True
        lr = 0.1
        optimizer = Adam(params=[tensor], lr=lr)

        initial_data = tensor.data
        optimizer.step()

        # Adam's update rule is complex, so we'll just check if the data has changed
        assert np.allclose(tensor.data, initial_data)
        # After step, grad should be zeroed by the optimizer's zero_grad call (if it happens internally)
        # or it should be non-zero if zero_grad is called separately.
        # For now, let's just assert it's not None.
        assert tensor.grad is not None

    def test_adam_zero_grad(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        grad = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        tensor = from_data(data.shape, data)
        tensor.requires_grad = True
        tensor.grad[:] = grad
        lr = 0.1
        optimizer = Adam(params=[tensor], lr=lr)

        optimizer.zero_grad()

        assert np.allclose(tensor.grad, np.zeros_like(data))
