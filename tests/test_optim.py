import pytest
import numpy as np
from idrak.core.tensor import Tensor
from idrak.optim.sgd import SGD
from idrak.optim.adam import Adam
from idrak.functions import from_data, zeros

# Helper to initialize a Tensor's data directly for testing purposes.
def _init_tensor_data(tensor: Tensor, data: np.ndarray):
    flat_data = data.flatten().astype(np.float32)
    for i, val in enumerate(flat_data):
        tensor.c_tensor_ptr.contents.data.contents.data[i] = ctypes.c_float(val)
"""
class TestOptimizers:

    # ======== SGD Optimizer ========
    def test_sgd_step(self):
        # Simple linear model: y = x * w
        x_np = np.array([[2.0]], dtype=np.float32)
        w_np = np.array([[3.0]], dtype=np.float32)
        target_y_np = np.array([[10.0]], dtype=np.float32)

        x = from_data(x_np.shape, x_np, requires_grad=False)
        w = from_data(w_np.shape, w_np, requires_grad=True)
        target_y = from_data(target_y_np.shape, target_y_np, requires_grad=False)

        optimizer = SGD(params=[w], lr=0.1)

        # Simulate one training step
        pred_y = x * w
        loss = (pred_y - target_y) ** 2 # MSE loss
        loss.backward()

        # Expected gradient for w: dL/dw = 2 * (pred_y - target_y) * x
        # pred_y = 2 * 3 = 6
        # loss = (6 - 10)^2 = 16
        # dL/dw = 2 * (6 - 10) * 2 = 2 * (-4) * 2 = -16
        assert np.allclose(w.grad, np.array([[-16.0]], dtype=np.float32))

        optimizer.step()

        # Expected w after step: w_new = w_old - lr * grad
        # w_new = 3.0 - 0.1 * (-16.0) = 3.0 + 1.6 = 4.6
        expected_w_after_step = np.array([[4.6]], dtype=np.float32)
        assert np.allclose(w.data, expected_w_after_step)

    def test_sgd_zero_grad(self):
        w = from_data((2, 2), np.ones((2, 2), dtype=np.float32), requires_grad=True)
        optimizer = SGD(params=[w], lr=0.1)

        # Manually set some gradients
        w.c_tensor_ptr.contents.grad.contents.data[0] = ctypes.c_float(5.0)
        w.c_tensor_ptr.contents.grad.contents.data[1] = ctypes.c_float(10.0)

        assert np.any(w.grad != 0.0)

        optimizer.zero_grad()
        assert np.all(w.grad == 0.0)

    # ======== Adam Optimizer ========
    def test_adam_step(self):
        # Simple linear model: y = x * w
        x_np = np.array([[2.0]], dtype=np.float32)
        w_np = np.array([[3.0]], dtype=np.float32)
        target_y_np = np.array([[10.0]], dtype=np.float32)

        x = from_data(x_np.shape, x_np, requires_grad=False)
        w = from_data(w_np.shape, w_np, requires_grad=True)
        target_y = from_data(target_y_np.shape, target_y_np, requires_grad=False)

        optimizer = Adam(params=[w], lr=0.1, betas=(0.9, 0.999), epsilon=1e-8)

        # Simulate one training step
        pred_y = x * w
        loss = (pred_y - target_y) ** 2 # MSE loss
        loss.backward()

        # Expected gradient for w: dL/dw = 2 * (pred_y - target_y) * x = -16
        assert np.allclose(w.grad, np.array([[-16.0]], dtype=np.float32))

        optimizer.step()

        # Adam update is complex, so we'll verify it moves in the right direction
        # and that the value changes. Exact value comparison is hard without re-implementing Adam.
        assert not np.allclose(w.data, w_np) # Should have changed
        # For a negative gradient, the weight should increase
        assert w.data[0,0] > w_np[0,0]

        # Simulate a second step
        optimizer.zero_grad()
        pred_y = x * w
        loss = (pred_y - target_y) ** 2
        loss.backward()
        optimizer.step()

        # Verify time_step increments
        assert optimizer.time_step == 3 # Initialized to 1, incremented twice

    def test_adam_zero_grad(self):
        w = from_data((2, 2), np.ones((2, 2), dtype=np.float32), requires_grad=True)
        optimizer = Adam(params=[w], lr=0.1)

        # Manually set some gradients
        w.c_tensor_ptr.contents.grad.contents.data[0] = ctypes.c_float(5.0)
        w.c_tensor_ptr.contents.grad.contents.data[1] = ctypes.c_float(10.0)

        assert np.any(w.grad != 0.0)

        optimizer.zero_grad()
        assert np.all(w.grad == 0.0)
"""
