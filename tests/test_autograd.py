import pytest
import numpy as np
from idrak.core.tensor import Tensor

def test_backward_add():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    a_idrak = Tensor(data=a_np, shape=(2, 2), requires_grad=True)
    b_idrak = Tensor(data=b_np, shape=(2, 2), requires_grad=True)

    c_idrak = a_idrak + b_idrak
    c_idrak.backward()

    # For z = x + y, dz/dx = 1, dz/dy = 1
    # So, dL/dx = dL/dz * dz/dx = dL/dz
    # And dL/dy = dL/dz * dz/dy = dL/dz
    # Since c_idrak is the final output, its gradient is initialized to ones.
    # Therefore, a_idrak.grad and b_idrak.grad should be all ones with the same shape.
    expected_grad_a = np.ones_like(a_np)
    expected_grad_b = np.ones_like(b_np)

    assert a_idrak.grad is not None
    assert b_idrak.grad is not None

    assert a_idrak.grad.shape == expected_grad_a.shape
    assert b_idrak.grad.shape == expected_grad_b.shape

    assert np.allclose(a_idrak.grad, expected_grad_a)
    assert np.allclose(b_idrak.grad, expected_grad_b)

def test_backward_sub():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    a_idrak = Tensor(data=a_np, shape=(2, 2), requires_grad=True)
    b_idrak = Tensor(data=b_np, shape=(2, 2), requires_grad=True)

    c_idrak = a_idrak - b_idrak
    c_idrak.backward()

    expected_grad_a = np.ones_like(a_np)
    expected_grad_b = -np.ones_like(b_np)

    assert a_idrak.grad is not None
    assert b_idrak.grad is not None

    assert np.allclose(a_idrak.grad, expected_grad_a)
    assert np.allclose(b_idrak.grad, expected_grad_b)

def test_backward_mul():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    a_idrak = Tensor(data=a_np, shape=(2, 2), requires_grad=True)
    b_idrak = Tensor(data=b_np, shape=(2, 2), requires_grad=True)

    c_idrak = a_idrak * b_idrak
    c_idrak.backward()

    expected_grad_a = b_np # dL/da = dL/dc * dc/da = 1 * b
    expected_grad_b = a_np # dL/db = dL/dc * dc/db = 1 * a

    assert a_idrak.grad is not None
    assert b_idrak.grad is not None

    assert np.allclose(a_idrak.grad, expected_grad_a)
    assert np.allclose(b_idrak.grad, expected_grad_b)

def test_backward_div():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    a_idrak = Tensor(data=a_np, shape=(2, 2), requires_grad=True)
    b_idrak = Tensor(data=b_np, shape=(2, 2), requires_grad=True)

    c_idrak = a_idrak / b_idrak
    c_idrak.backward()

    expected_grad_a = 1 / b_np # dL/da = dL/dc * dc/da = 1 * (1/b)
    expected_grad_b = -a_np / (b_np ** 2) # dL/db = dL/dc * dc/db = 1 * (-a/b^2)

    assert a_idrak.grad is not None
    assert b_idrak.grad is not None

    assert np.allclose(a_idrak.grad, expected_grad_a)
    assert np.allclose(b_idrak.grad, expected_grad_b)

def test_backward_neg():
    a_np = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2, 2), requires_grad=True)

    b_idrak = -a_idrak
    b_idrak.backward()

    expected_grad_a = -np.ones_like(a_np) # dL/da = dL/db * db/da = 1 * -1

    assert a_idrak.grad is not None
    assert np.allclose(a_idrak.grad, expected_grad_a)

def test_backward_relu():
    a_np = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2, 2), requires_grad=True)

    b_idrak = a_idrak.relu()
    b_idrak.backward()

    expected_grad_a = (a_np > 0).astype(np.float32) # dL/da = dL/db * db/da = 1 * (1 if a > 0 else 0)

    assert a_idrak.grad is not None
    assert np.allclose(a_idrak.grad, expected_grad_a)

def test_backward_log():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2, 2), requires_grad=True)

    b_idrak = a_idrak.log()
    b_idrak.backward()

    expected_grad_a = 1 / a_np # dL/da = dL/db * db/da = 1 * (1/a)

    assert a_idrak.grad is not None
    assert np.allclose(a_idrak.grad, expected_grad_a)

def test_backward_exp():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2, 2), requires_grad=True)

    b_idrak = a_idrak.exp()
    b_idrak.backward()

    expected_grad_a = np.exp(a_np) # dL/da = dL/db * db/da = 1 * exp(a)

    assert a_idrak.grad is not None
    assert np.allclose(a_idrak.grad, expected_grad_a)

def test_backward_abs():
    a_np = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2, 2), requires_grad=True)

    b_idrak = a_idrak.abs()
    b_idrak.backward()

    expected_grad_a = np.sign(a_np) # dL/da = dL/db * db/da = 1 * sign(a)

    assert a_idrak.grad is not None
    assert np.allclose(a_idrak.grad, expected_grad_a)

def test_backward_matmul():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    a_idrak = Tensor(data=a_np, shape=(2, 2), requires_grad=True)
    b_idrak = Tensor(data=b_np, shape=(2, 2), requires_grad=True)

    c_idrak = a_idrak @ b_idrak
    c_idrak.backward()

    # dL/dA = dL/dC @ B.T
    # dL/dB = A.T @ dL/dC
    # Since dL/dC is initialized to ones_like(C)
    expected_grad_a = np.ones_like(c_idrak.data) @ b_np.T
    expected_grad_b = a_np.T @ np.ones_like(c_idrak.data)

    assert a_idrak.grad is not None
    assert b_idrak.grad is not None

    assert np.allclose(a_idrak.grad, expected_grad_a)
    assert np.allclose(b_idrak.grad, expected_grad_b)

def test_backward_full_graph():
    # Graph: d = relu( (a * b) + c )
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
    c_np = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)

    a_idrak = Tensor(data=a_np, shape=(2, 2), requires_grad=True)
    b_idrak = Tensor(data=b_np, shape=(2, 2), requires_grad=True)
    c_idrak = Tensor(data=c_np, shape=(2, 2), requires_grad=True)

    # Forward pass
    x_idrak = a_idrak * b_idrak
    y_idrak = x_idrak + c_idrak
    d_idrak = y_idrak.relu()

    # Backward pass
    d_idrak.backward()

    # Manual gradient calculation using chain rule
    # dL/dd = 1 (initialized by backward())

    # dL/dy = dL/dd * dd/dy
    # dd/dy for ReLU is 1 where y > 0, else 0
    y_np = (a_np * b_np) + c_np
    dL_dy = (y_np > 0).astype(np.float32)

    # dL/dx = dL/dy * dy/dx
    # dy/dx for add is 1
    dL_dx = dL_dy

    # dL/dc = dL/dy * dy/dc
    # dy/dc for add is 1
    dL_dc = dL_dy

    # dL/da = dL/dx * dx/da
    # dx/da for mul is b
    dL_da = dL_dx * b_np

    # dL/db = dL/dx * dx/db
    # dx/db for mul is a
    dL_db = dL_dx * a_np

    assert a_idrak.grad is not None
    assert b_idrak.grad is not None
    assert c_idrak.grad is not None

    assert np.allclose(a_idrak.grad, dL_da)
    assert np.allclose(b_idrak.grad, dL_db)
    assert np.allclose(c_idrak.grad, dL_dc)


