import pytest
import numpy as np
from idrak.core.tensor import Tensor
from idrak.nn.linear import Linear
from idrak.optim.sgd import SGD
from idrak.optim.adam import Adam

def test_sgd_init():
    param1 = Tensor(data=np.array([1.0]), shape=(1,), requires_grad=True)
    param2 = Tensor(data=np.array([2.0]), shape=(1,), requires_grad=True)
    params = [param1, param2]
    lr = 0.01

    optimizer = SGD(params, lr)

    assert len(optimizer.params) == 2
    assert optimizer.num_params == 2
    assert optimizer.lr == lr

def test_sgd_zero_grad():
    param1 = Tensor(data=np.array([1.0]), shape=(1,), requires_grad=True)
    param2 = Tensor(data=np.array([2.0]), shape=(1,), requires_grad=True)
    params = [param1, param2]
    lr = 0.01

    optimizer = SGD(params, lr)

    # Manually set some gradients (this part is removed as grad is read-only)
    # param1.grad = Tensor(data=np.array([0.5]), shape=(1,))
    # param2.grad = Tensor(data=np.array([0.8]), shape=(1,))

    # To make sure there are gradients to clear, we'll perform a dummy backward pass
    dummy_output = param1 * param2
    dummy_output.backward()

    assert param1.grad is not None
    assert param2.grad is not None

    optimizer.zero_grad()

    # Gradients should be None after zero_grad
    assert np.allclose(param1.grad, np.zeros_like(param1.data))
    assert np.allclose(param2.grad, np.zeros_like(param2.data))

def test_sgd_step():
    # Simple linear model: y = wx + b
    # Loss: MSE = (y_pred - y_true)^2
    # dL/dw = 2 * (y_pred - y_true) * x
    # dL/db = 2 * (y_pred - y_true)

    # Initialize parameters
    w_np = np.array([[0.5]], dtype=np.float32)
    b_np = np.array([[0.1]], dtype=np.float32)
    w = Tensor(data=w_np, shape=(1, 1), requires_grad=True)
    b = Tensor(data=b_np, shape=(1, 1), requires_grad=True)

    # Optimizer
    lr = 0.01
    optimizer = SGD([w, b], lr)

    # Data
    x_np = np.array([[2.0]], dtype=np.float32)
    y_true_np = np.array([[1.5]], dtype=np.float32)
    x = Tensor(data=x_np, shape=(1, 1))
    y_true = Tensor(data=y_true_np, shape=(1, 1))

    # Forward pass
    y_pred = x @ w + b
    y_pred.realize()

    # Loss (MSE)
    loss = (y_pred - y_true) * (y_pred - y_true)
    loss.realize()

    # Backward pass
    loss.backward()

    # Store initial parameters and gradients
    initial_w_data = w.data.copy()
    initial_b_data = b.data.copy()
    initial_w_grad = np.array(w.grad.data).copy()
    initial_b_grad = np.array(b.grad.data).copy()

    # Perform optimization step
    optimizer.step()

    # Verify parameter updates
    # w_new = w_old - lr * dL/dw
    # b_new = b_old - lr * dL/db
    expected_w_new = initial_w_data - lr * initial_w_grad
    expected_b_new = initial_b_data - lr * initial_b_grad

    w.realize()
    b.realize()

    assert np.allclose(w.data, expected_w_new, rtol=1e-4, atol=1e-6)
    assert np.allclose(b.data, expected_b_new, rtol=1e-4, atol=1e-6)

    # Gradients should be cleared after step (due to zero_grad call within step or implicitly)
    # Note: idrak's zero_grad sets grad to None, so we check for None
    optimizer.zero_grad() # Explicitly call zero_grad to ensure it's cleared for next iteration
    assert np.allclose(w.grad, np.zeros_like(w.data))
    assert np.allclose(b.grad, np.zeros_like(b.data))

def test_adam_init():
    param1 = Tensor(data=np.array([1.0]), shape=(1,), requires_grad=True)
    param2 = Tensor(data=np.array([2.0]), shape=(1,), requires_grad=True)
    params = [param1, param2]
    lr = 0.001
    betas = (0.9, 0.999)
    epsilon = 1e-8

    optimizer = Adam(params, lr, betas, epsilon)

    assert len(optimizer.params) == 2
    assert optimizer.num_params == 2
    assert optimizer.lr == lr
    assert optimizer.betas == betas
    assert optimizer.epsilon == epsilon
    assert optimizer.time_step == 1

    # Check mt and vt initialization
    assert len(optimizer._mt_tensors) == 2
    assert len(optimizer._vt_tensors) == 2
    for mt_t, vt_t, param in zip(optimizer._mt_tensors, optimizer._vt_tensors, params):
        mt_t.realize()
        vt_t.realize()
        assert mt_t.shape == param.shape
        assert vt_t.shape == param.shape
        assert np.allclose(mt_t.data, np.zeros_like(param.data))
        assert np.allclose(vt_t.data, np.zeros_like(param.data))

def test_adam_zero_grad():
    param1 = Tensor(data=np.array([1.0]), shape=(1,), requires_grad=True)
    param2 = Tensor(data=np.array([2.0]), shape=(1,), requires_grad=True)
    params = [param1, param2]
    lr = 0.001

    optimizer = Adam(params, lr)

    # To make sure there are gradients to clear, we'll perform a dummy backward pass
    dummy_output = param1 * param2
    dummy_output.backward()

    assert param1.grad is not None
    assert param2.grad is not None

    optimizer.zero_grad()

    assert np.allclose(param1.grad, np.zeros_like(param1.data))
    assert np.allclose(param2.grad, np.zeros_like(param2.data))

def test_adam_step():
    # This test will be less precise due to Adam's complex update rule
    # We'll check if parameters change and time_step increments

    w_np = np.array([[0.5]], dtype=np.float32)
    b_np = np.array([[0.1]], dtype=np.float32)
    w = Tensor(data=w_np, shape=(1, 1), requires_grad=True)
    b = Tensor(data=b_np, shape=(1, 1), requires_grad=True)

    lr = 0.001
    optimizer = Adam([w, b], lr)

    x_np = np.array([[2.0]], dtype=np.float32)
    y_true_np = np.array([[1.5]], dtype=np.float32)
    x = Tensor(data=x_np, shape=(1, 1))
    y_true = Tensor(data=y_true_np, shape=(1, 1))

    # Perform a few steps and check if parameters change
    initial_w_data = w.data.copy()
    initial_b_data = b.data.copy()

    for _ in range(5):
        # Forward pass
        y_pred = x @ w + b
        y_pred.realize()

        # Loss (MSE)
        loss = (y_pred - y_true) * (y_pred - y_true)
        loss.realize()

        # Backward pass
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    w.realize()
    b.realize()

    # Parameters should have changed
    assert not np.allclose(w.data, initial_w_data)
    assert not np.allclose(b.data, initial_b_data)

    # time_step should have incremented
    assert optimizer.time_step == 1 + 5 # Initial time_step is 1, plus 5 steps