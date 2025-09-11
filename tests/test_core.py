import pytest
import numpy as np
from idrak.core.tensor import Tensor

def test_tensor_creation():
    # Test creation from list
    data_list = [[1.0, 2.0], [3.0, 4.0]]
    t_list = Tensor(data=data_list, shape=(2, 2))
    assert t_list.shape == (2, 2)
    assert np.array_equal(t_list.data, np.array(data_list, dtype=np.float32))

    # Test creation from numpy array
    data_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    t_np = Tensor(data=data_np, shape=(2, 2))
    assert t_np.shape == (2, 2)
    assert np.array_equal(t_np.data, data_np)

    # Test scalar tensor
    scalar_data = 10.0
    t_scalar = Tensor(data=scalar_data, shape=())
    assert t_scalar.shape == ()
    assert np.array_equal(t_scalar.data, np.array(scalar_data, dtype=np.float32))

def test_tensor_add():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    a_idrak = Tensor(data=a_np, shape=(2, 2))
    b_idrak = Tensor(data=b_np, shape=(2, 2))

    c_idrak = a_idrak + b_idrak
    c_idrak.realize()
    c_np = a_np + b_np

    assert c_idrak.shape == c_np.shape
    assert np.allclose(c_idrak.data, c_np)

def test_tensor_add_scalar():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    scalar = 5.0

    a_idrak = Tensor(data=a_np, shape=(2, 2))

    c_idrak = a_idrak + scalar
    c_idrak.realize()
    c_np = a_np + scalar

    assert c_idrak.shape == c_np.shape
    assert np.allclose(c_idrak.data, c_np)

    c_idrak_r = scalar + a_idrak
    c_idrak_r.realize()
    assert c_idrak_r.shape == c_np.shape
    assert np.allclose(c_idrak_r.data, c_np)

def test_tensor_sub():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    a_idrak = Tensor(data=a_np, shape=(2, 2))
    b_idrak = Tensor(data=b_np, shape=(2, 2))

    c_idrak = a_idrak - b_idrak
    c_idrak.realize()
    c_np = a_np - b_np

    assert c_idrak.shape == c_np.shape
    assert np.allclose(c_idrak.data, c_np)

def test_tensor_mul():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    a_idrak = Tensor(data=a_np, shape=(2, 2))
    b_idrak = Tensor(data=b_np, shape=(2, 2))

    c_idrak = a_idrak * b_idrak
    c_idrak.realize()
    c_np = a_np * b_np

    assert c_idrak.shape == c_np.shape
    assert np.allclose(c_idrak.data, c_np)

def test_tensor_div():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    a_idrak = Tensor(data=a_np, shape=(2, 2))
    b_idrak = Tensor(data=b_np, shape=(2, 2))

    c_idrak = a_idrak / b_idrak
    c_idrak.realize()
    c_np = a_np / b_np

    assert c_idrak.shape == c_np.shape
    assert np.allclose(c_idrak.data, c_np)

def test_tensor_pow():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.array([[2.0, 3.0], [1.0, 2.0]], dtype=np.float32)

    a_idrak = Tensor(data=a_np, shape=(2, 2))
    b_idrak = Tensor(data=b_np, shape=(2, 2))

    c_idrak = a_idrak ** b_idrak
    c_idrak.realize()
    c_np = a_np ** b_np

    assert c_idrak.shape == c_np.shape
    assert np.allclose(c_idrak.data, c_np)

def test_tensor_matmul():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    a_idrak = Tensor(data=a_np, shape=(2, 2))
    b_idrak = Tensor(data=b_np, shape=(2, 2))

    c_idrak = a_idrak @ b_idrak
    c_idrak.realize()
    c_np = a_np @ b_np

    assert c_idrak.shape == c_np.shape
    assert np.allclose(c_idrak.data, c_np)

def test_tensor_neg():
    a_np = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2, 2))

    b_idrak = -a_idrak
    b_idrak.realize()
    b_np = -a_np

    assert b_idrak.shape == b_np.shape
    assert np.allclose(b_idrak.data, b_np)

def test_tensor_relu():
    a_np = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2, 2))

    b_idrak = a_idrak.relu()
    b_idrak.realize()
    b_np = np.maximum(0, a_np)

    assert b_idrak.shape == b_np.shape
    assert np.allclose(b_idrak.data, b_np)

def test_tensor_log():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2, 2))

    b_idrak = a_idrak.log()
    b_idrak.realize()
    b_np = np.log(a_np)

    assert b_idrak.shape == b_np.shape
    assert np.allclose(b_idrak.data, b_np)

def test_tensor_exp():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2, 2))

    b_idrak = a_idrak.exp()
    b_idrak.realize()
    b_np = np.exp(a_np)

    assert b_idrak.shape == b_np.shape
    assert np.allclose(b_idrak.data, b_np)

def test_tensor_abs():
    a_np = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2, 2))

    b_idrak = a_idrak.abs()
    b_idrak.realize()
    b_np = np.abs(a_np)

    assert b_idrak.shape == b_np.shape
    assert np.allclose(b_idrak.data, b_np)

def test_tensor_view():
    a_np = np.arange(1.0, 7.0).reshape(2, 3).astype(np.float32)
    a_idrak = Tensor(data=a_np, shape=(2, 3))

    b_idrak = a_idrak.view((3, 2))
    b_idrak.realize()
    b_np = a_np.view().reshape((3, 2))

    assert b_idrak.shape == b_np.shape
    assert np.allclose(b_idrak.data, b_np)

def test_tensor_unsqueeze():
    a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(3,))

    b_idrak = a_idrak.unsqueeze(dim=0)
    b_idrak.realize()
    b_np = np.expand_dims(a_np, axis=0)

    assert b_idrak.shape == b_np.shape
    assert np.allclose(b_idrak.data, b_np)

    b_idrak = a_idrak.unsqueeze(dim=1)
    b_idrak.realize()
    b_np = np.expand_dims(a_np, axis=1)

    assert b_idrak.shape == b_np.shape
    assert np.allclose(b_idrak.data, b_np)

def test_tensor_squeeze():
    a_np = np.array([[[1.0, 2.0]]], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(1, 1, 2))

    b_idrak = a_idrak.squeeze(dim=0)
    b_idrak.realize()
    b_np = np.squeeze(a_np, axis=0)

    assert b_idrak.shape == b_np.shape
    assert np.allclose(b_idrak.data, b_np)

    a_np_2 = np.array([[[1.0, 2.0]]], dtype=np.float32)
    a_idrak_2 = Tensor(data=a_np_2, shape=(1, 1, 2))

    b_idrak_2 = a_idrak_2.squeeze(dim=1)
    b_idrak_2.realize()
    b_np_2 = np.squeeze(a_np_2, axis=1)

    assert b_idrak_2.shape == b_np_2.shape
    assert np.allclose(b_idrak_2.data, b_np_2)

def test_tensor_expand():
    a_np = np.array([1.0, 2.0], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(1, 2))

    b_idrak = a_idrak.expand((2, 2))
    b_idrak.realize()
    b_np = np.broadcast_to(a_np, (2, 2))
    print(b_idrak)

    assert b_idrak.shape == b_np.shape
    assert np.allclose(b_idrak.data, b_np)

def test_tensor_broadcast():
    a_np = np.array([1.0, 2.0], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2,))

    b_idrak = a_idrak.broadcast((2, 2))
    b_idrak.realize()
    b_np = np.broadcast_to(a_np, (2, 2))

    assert b_idrak.shape == b_np.shape
    assert np.allclose(b_idrak.data, b_np)

def test_tensor_transpose():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2, 2))

    b_idrak = a_idrak.transpose(0, 1)
    b_idrak.realize()
    b_np = a_np.T

    assert b_idrak.shape == b_np.shape
    assert np.allclose(b_idrak.data, b_np)

def test_tensor_sum():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2, 2))

    # Sum all elements
    b_idrak = a_idrak.sum()
    b_idrak.realize()
    b_np = np.sum(a_np)
    assert b_idrak.shape == (1,)
    assert np.allclose(b_idrak.item(), b_np)

    # Sum along dim 0
    b_idrak_dim0 = a_idrak.sum(dim=0, keepdim=False)
    b_idrak_dim0.realize()
    b_np_dim0 = np.sum(a_np, axis=0)
    print(b_idrak_dim0, b_np_dim0)
    assert b_idrak_dim0.shape == b_np_dim0.shape
    assert np.allclose(b_idrak_dim0.data, b_np_dim0)

    # Sum along dim 1, keepdim=True
    b_idrak_dim1_keepdim = a_idrak.sum(dim=1, keepdim=True)
    b_idrak_dim1_keepdim.realize()
    b_np_dim1_keepdim = np.sum(a_np, axis=1, keepdims=True)
    assert b_idrak_dim1_keepdim.shape == b_np_dim1_keepdim.shape
    assert np.allclose(b_idrak_dim1_keepdim.data, b_np_dim1_keepdim)

def test_tensor_mean():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2, 2))

    # Mean all elements
    b_idrak = a_idrak.mean()
    b_idrak.realize()
    b_np = np.mean(a_np)
    assert b_idrak.shape == (1,)
    assert np.allclose(b_idrak.data.item(), b_np.item())

    # Mean along dim 0
    b_idrak_dim0 = a_idrak.mean(dim=0, keepdim=False)
    b_idrak_dim0.realize()
    b_np_dim0 = np.mean(a_np, axis=0)
    assert b_idrak_dim0.shape == b_np_dim0.shape
    assert np.allclose(b_idrak_dim0.data, b_np_dim0)

    # Mean along dim 1, keepdim=True
    b_idrak_dim1_keepdim = a_idrak.mean(dim=1, keepdim=True)
    b_idrak_dim1_keepdim.realize()
    b_np_dim1_keepdim = np.mean(a_np, axis=1, keepdims=True)
    assert b_idrak_dim1_keepdim.shape == b_np_dim1_keepdim.shape
    assert np.allclose(b_idrak_dim1_keepdim.data, b_np_dim1_keepdim)

def test_tensor_max():
    a_np = np.array([[1.0, 5.0], [3.0, 2.0]], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2, 2))

    # Max all elements
    b_idrak = a_idrak.max()
    b_idrak.realize()
    b_np = np.max(a_np)
    assert b_idrak.shape == (1,)
    assert np.allclose(b_idrak.data.item(), b_np.item())

    # Max along dim 0
    b_idrak_dim0 = a_idrak.max(dim=0, keepdim=False)
    b_idrak_dim0.realize()
    b_np_dim0 = np.max(a_np, axis=0)
    assert b_idrak_dim0.shape == b_np_dim0.shape
    assert np.allclose(b_idrak_dim0.data, b_np_dim0)

    # Max along dim 1, keepdim=True
    b_idrak_dim1_keepdim = a_idrak.max(dim=1, keepdim=True)
    b_idrak_dim1_keepdim.realize()
    b_np_dim1_keepdim = np.max(a_np, axis=1, keepdims=True)
    assert b_idrak_dim1_keepdim.shape == b_np_dim1_keepdim.shape
    assert np.allclose(b_idrak_dim1_keepdim.data, b_np_dim1_keepdim)

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
