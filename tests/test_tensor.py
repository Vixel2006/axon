import pyteset
import numpy as np
import os
import sys

from py.core.tensor import Tensor

# Helper for numerical gradient checking
def numerical_gradient(func, x, delta=1e-4):
    grad = np.zeros_like(x.data)
    it = np.nditer(x.data, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        original_value = x.data[idx]

        # f(x + delta)
        x.data[idx] = original_value + delta
        f_plus_delta = func().data

        # f(x - delta)
        x.data[idx] = original_value - delta
        f_minus_delta = func().data

        # Restore original value
        x.data[idx] = original_value

        grad[idx] = (f_plus_delta - f_minus_delta) / (2 * delta)
        it.iternext()
    return grad

# Test Tensor initialization
def test_tensor_init_empty():
    t = Tensor()
    assert t.shape == ()
    assert t.ndim == 0
    assert t.data == np.array(0.0, dtype=np.float32) # Default empty tensor is scalar 0.0

def test_tensor_init_shape_only():
    t = Tensor(shape=(2, 3))
    assert t.shape == (2, 3)
    assert t.ndim == 2
    assert t.data.shape == (2, 3)
    assert np.allclose(t.data, np.zeros((2, 3), dtype=np.float32)) # Default to zeros

def test_tensor_init_with_data():
    data = [[1.0, 2.0], [3.0, 4.0]]
    t = Tensor(shape=(2, 2), data=data)
    assert t.shape == (2, 2)
    assert t.ndim == 2
    assert np.allclose(t.data, np.array(data, dtype=np.float32))

def test_tensor_init_scalar():
    t = Tensor(shape=(), data=5.0)
    assert t.shape == ()
    assert t.ndim == 0
    assert np.allclose(t.data, np.array(5.0, dtype=np.float32))

# Test properties
def test_tensor_properties():
    t = Tensor(shape=(2, 2), data=[[1, 2], [3, 4]], requires_grad=True)
    assert t.shape == (2, 2)
    assert t.ndim == 2
    assert t.requires_grad is True
    assert t.grad is None # Grad should be None initially

    t_no_grad = Tensor(shape=(2, 2), data=[[1, 2], [3, 4]], requires_grad=False)
    assert t_no_grad.requires_grad is False

# Test basic addition
def test_tensor_add():
    a = Tensor(shape=(2, 2), data=[[1, 2], [3, 4]])
    b = Tensor(shape=(2, 2), data=[[5, 6], [7, 8]])
    c = a + b
    c.realize()
    expected_c = np.array([[6, 8], [10, 12]], dtype=np.float32)
    assert np.allclose(c.data, expected_c)

def test_tensor_add_scalar():
    a = Tensor(shape=(2, 2), data=[[1, 2], [3, 4]])
    b = 5.0
    c = a + b
    c.realize()
    expected_c = np.array([[6, 7], [8, 9]], dtype=np.float32)
    assert np.allclose(c.data, expected_c)

def test_tensor_add_scalar_radd():
    a = 5.0
    b = Tensor(shape=(2, 2), data=[[1, 2], [3, 4]])
    c = a + b
    c.realize()
    expected_c = np.array([[6, 7], [8, 9]], dtype=np.float32)
    assert np.allclose(c.data, expected_c)

# Test basic addition backward
def test_tensor_add_backward():
    a = Tensor(shape=(2, 2), data=[[1, 2], [3, 4]], requires_grad=True)
    b = Tensor(shape=(2, 2), data=[[5, 6], [7, 8]], requires_grad=True)
    c = a + b
    c.backward()

    # For c = a + b, dC/dA = 1, dC/dB = 1. So grad should be ones.
    assert np.allclose(a.grad, np.ones((2, 2), dtype=np.float32))
    assert np.allclose(b.grad, np.ones((2, 2), dtype=np.float32))

def test_tensor_add_scalar_backward():
    a = Tensor(shape=(2, 2), data=[[1, 2], [3, 4]], requires_grad=True)
    b_scalar = 5.0
    c = a + b_scalar
    c.backward()
    # For c = a + scalar, dC/dA = 1.
    assert np.allclose(a.grad, np.ones((2, 2), dtype=np.float32))

# Test basic subtraction
def test_tensor_sub():
    a = Tensor(shape=(2, 2), data=[[1, 2], [3, 4]])
    b = Tensor(shape=(2, 2), data=[[5, 6], [7, 8]])
    c = a - b
    c.realize()
    expected_c = np.array([[-4, -4], [-4, -4]], dtype=np.float32)
    assert np.allclose(c.data, expected_c)

def test_tensor_sub_scalar():
    a = Tensor(shape=(2, 2), data=[[1, 2], [3, 4]])
    b = 5.0
    c = a - b
    c.realize()
    expected_c = np.array([[-4, -3], [-2, -1]], dtype=np.float32)
    assert np.allclose(c.data, expected_c)

def test_tensor_sub_scalar_rsub():
    a = 5.0
    b = Tensor(shape=(2, 2), data=[[1, 2], [3, 4]])
    c = a - b
    c.realize()
    expected_c = np.array([[4, 3], [2, 1]], dtype=np.float32)
    assert np.allclose(c.data, expected_c)

# Test basic subtraction backward
def test_tensor_sub_backward():
    a = Tensor(shape=(2, 2), data=[[1, 2], [3, 4]], requires_grad=True)
    b = Tensor(shape=(2, 2), data=[[5, 6], [7, 8]], requires_grad=True)
    c = a - b
    c.backward()

    # For c = a - b, dC/dA = 1, dC/dB = -1.
    assert np.allclose(a.grad, np.ones((2, 2), dtype=np.float32))
    assert np.allclose(b.grad, -np.ones((2, 2), dtype=np.float32))

def test_tensor_sub_scalar_backward():
    a = Tensor(shape=(2, 2), data=[[1, 2], [3, 4]], requires_grad=True)
    b_scalar = 5.0
    c = a - b_scalar
    c.backward()
    # For c = a - scalar, dC/dA = 1.
    assert np.allclose(a.grad, np.ones((2, 2), dtype=np.float32))

def test_tensor_rsub_scalar_backward():
    a_scalar = 5.0
    b = Tensor(shape=(2, 2), data=[[1, 2], [3, 4]], requires_grad=True)
    c = a_scalar - b
    c.backward()
    # For c = scalar - b, dC/dB = -1.
    assert np.allclose(b.grad, -np.ones((2, 2), dtype=np.float32))

# Test basic multiplication
def test_tensor_mul():
    a = Tensor(shape=(2, 2), data=[[1, 2], [3, 4]])
    b = Tensor(shape=(2, 2), data=[[5, 6], [7, 8]])
    c = a * b
    c.realize()
    expected_c = np.array([[5, 12], [21, 32]], dtype=np.float32)
    assert np.allclose(c.data, expected_c)

def test_tensor_mul_scalar():
    a = Tensor(shape=(2, 2), data=[[1, 2], [3, 4]])
    b = 5.0
    c = a * b
    c.realize()
    expected_c = np.array([[5, 10], [15, 20]], dtype=np.float32)
    assert np.allclose(c.data, expected_c)

def test_tensor_mul_scalar_rmul():
    a = 5.0
    b = Tensor(shape=(2, 2), data=[[1, 2], [3, 4]])
    c = a * b
    c.realize()
    expected_c = np.array([[5, 10], [15, 20]], dtype=np.float32)
    assert np.allclose(c.data, expected_c)

# Test basic multiplication backward
def test_tensor_mul_backward():
    a_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b_data = np.array([[5, 6], [7, 8]], dtype=np.float32)
    a = Tensor(shape=(2, 2), data=a_data, requires_grad=True)
    b = Tensor(shape=(2, 2), data=b_data, requires_grad=True)
    c = a * b
    c.backward()

    # For c = a * b, dC/dA = b, dC/dB = a.
    assert np.allclose(a.grad, b_data)
    assert np.allclose(b.grad, a_data)

def test_tensor_mul_scalar_backward():
    a_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    a = Tensor(shape=(2, 2), data=a_data, requires_grad=True)
    b_scalar = 5.0
    c = a * b_scalar
    c.backward()
    # For c = a * scalar, dC/dA = scalar.
    assert np.allclose(a.grad, np.full_like(a_data, b_scalar))

# Test basic division
def test_tensor_div():
    a = Tensor(shape=(2, 2), data=[[10, 20], [30, 40]])
    b = Tensor(shape=(2, 2), data=[[2, 5], [6, 8]])
    c = a / b
    c.realize()
    expected_c = np.array([[5, 4], [5, 5]], dtype=np.float32)
    assert np.allclose(c.data, expected_c)

def test_tensor_div_scalar():
    a = Tensor(shape=(2, 2), data=[[10, 20], [30, 40]])
    b = 5.0
    c = a / b
    c.realize()
    expected_c = np.array([[2, 4], [6, 8]], dtype=np.float32)
    assert np.allclose(c.data, expected_c)

def test_tensor_div_scalar_rdiv():
    a = 10.0
    b = Tensor(shape=(2, 2), data=[[1, 2], [5, 10]])
    c = a / b
    c.realize()
    expected_c = np.array([[10, 5], [2, 1]], dtype=np.float32)
    assert np.allclose(c.data, expected_c)

# Test basic division backward
def test_tensor_div_backward():
    a_data = np.array([[10, 20], [30, 40]], dtype=np.float32)
    b_data = np.array([[2, 5], [6, 8]], dtype=np.float32)
    a = Tensor(shape=(2, 2), data=a_data, requires_grad=True)
    b = Tensor(shape=(2, 2), data=b_data, requires_grad=True)
    c = a / b
    c.backward()

    # For c = a / b, dC/dA = 1/b, dC/dB = -a/b^2.
    assert np.allclose(a.grad, 1 / b_data)
    assert np.allclose(b.grad, -a_data / (b_data ** 2))

def test_tensor_div_scalar_backward():
    a_data = np.array([[10, 20], [30, 40]], dtype=np.float32)
    a = Tensor(shape=(2, 2), data=a_data, requires_grad=True)
    b_scalar = 5.0
    c = a / b_scalar
    c.backward()
    # For c = a / scalar, dC/dA = 1/scalar.
    assert np.allclose(a.grad, np.full_like(a_data, 1 / b_scalar))

def test_tensor_rdiv_scalar_backward():
    a_scalar = 10.0
    b_data = np.array([[1, 2], [5, 10]], dtype=np.float32)
    b = Tensor(shape=(2, 2), data=b_data, requires_grad=True)
    c = a_scalar / b
    c.backward()
    # For c = scalar / b, dC/dB = -scalar/b^2.
    assert np.allclose(b.grad, -a_scalar / (b_data ** 2))

# Test matrix multiplication
def test_tensor_matmul():
    a = Tensor(shape=(2, 3), data=[[1, 2, 3], [4, 5, 6]])
    b = Tensor(shape=(3, 2), data=[[7, 8], [9, 10], [11, 12]])
    c = a @ b
    c.realize()
    expected_c = np.array([[58, 64], [139, 154]], dtype=np.float32)
    assert np.allclose(c.data, expected_c)

def test_tensor_matmul_backward():
    a_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    b_data = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32)
    a = Tensor(shape=(2, 3), data=a_data, requires_grad=True)
    b = Tensor(shape=(3, 2), data=b_data, requires_grad=True)
    c = a @ b
    c.backward()

    # dC/dA = grad_output @ B.T
    # dC/dB = A.T @ grad_output
    # Since grad_output is initialized to ones for the output tensor
    grad_output = np.ones_like(c.data)
    expected_a_grad = grad_output @ b_data.T
    expected_b_grad = a_data.T @ grad_output

    assert np.allclose(a.grad, expected_a_grad)
    assert np.allclose(b.grad, expected_b_grad)

# Test negation
def test_tensor_neg():
    a = Tensor(shape=(2, 2), data=[[1, -2], [3, -4]])
    b = -a
    b.realize()
    expected_b = np.array([[-1, 2], [-3, 4]], dtype=np.float32)
    assert np.allclose(b.data, expected_b)

def test_tensor_neg_backward():
    a_data = np.array([[1, -2], [3, -4]], dtype=np.float32)
    a = Tensor(shape=(2, 2), data=a_data, requires_grad=True)
    b = -a
    b.backward()

    # For b = -a, dB/dA = -1.
    assert np.allclose(a.grad, -np.ones_like(a_data))

# Test sum reduction
def test_tensor_sum():
    a = Tensor(shape=(2, 3), data=[[1, 2, 3], [4, 5, 6]])
    s = a.sum(axis=0, keepdim=False)
    s.realize()
    expected_s = np.array([5, 7, 9], dtype=np.float32)
    assert np.allclose(s.data, expected_s)

    s_keepdim = a.sum(axis=1, keepdim=True)
    s_keepdim.realize()
    expected_s_keepdim = np.array([[6], [15]], dtype=np.float32)
    assert np.allclose(s_keepdim.data, expected_s_keepdim)

    s_all = a.sum(axis=None, keepdim=False) # Sum all elements
    s_all.realize()
    expected_s_all = np.array(21.0, dtype=np.float32)
    assert np.allclose(s_all.data, expected_s_all)

def test_tensor_sum_backward():
    a_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    a = Tensor(shape=(2, 3), data=a_data, requires_grad=True)
    s = a.sum(axis=0, keepdim=False)
    s.backward()

    # For sum, gradient is 1 for each element that contributed to the sum.
    # If sum(axis=0), then each column sums to 1.
    expected_a_grad = np.array([[1, 1, 1], [1, 1, 1]], dtype=np.float32)
    assert np.allclose(a.grad, expected_a_grad)

    a_data_2 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    a2 = Tensor(shape=(2, 2), data=a_data_2, requires_grad=True)
    s2 = a2.sum(axis=None, keepdim=False) # Sum all
    s2.backward()
    assert np.allclose(a2.grad, np.ones_like(a_data_2))

# Test mean reduction
def test_tensor_mean():
    a = Tensor(shape=(2, 3), data=[[1, 2, 3], [4, 5, 6]])
    m = a.mean(axis=0, keepdim=False)
    m.realize()
    expected_m = np.array([2.5, 3.5, 4.5], dtype=np.float32)
    assert np.allclose(m.data, expected_m)

    m_keepdim = a.mean(axis=1, keepdim=True)
    m_keepdim.realize()
    expected_m_keepdim = np.array([[2.0], [5.0]], dtype=np.float32)
    assert np.allclose(m_keepdim.data, expected_m_keepdim)

    m_all = a.mean(axis=None, keepdim=False) # Mean all elements
    m_all.realize()
    expected_m_all = np.array(21.0 / 6.0, dtype=np.float32)
    assert np.allclose(m_all.data, expected_m_all)

def test_tensor_mean_backward():
    a_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    a = Tensor(shape=(2, 3), data=a_data, requires_grad=True)
    m = a.mean(axis=0, keepdim=False)
    m.backward()

    # For mean(axis=0), gradient is 1/N where N is the size of the axis being reduced.
    # Here, N=2 (rows).
    expected_a_grad = np.full_like(a_data, 1/2.0)
    assert np.allclose(a.grad, expected_a_grad)

    a_data_2 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    a2 = Tensor(shape=(2, 2), data=a_data_2, requires_grad=True)
    m2 = a2.mean(axis=None, keepdim=False) # Mean all
    m2.backward()
    # N=4 (total elements)
    assert np.allclose(a2.grad, np.full_like(a_data_2, 1/4.0))

# Test max reduction (forward only for now, backward is complex)
def test_tensor_max():
    a = Tensor(shape=(2, 3), data=[[1, 5, 3], [4, 2, 6]])
    m = a.max(axis=0, keepdim=False)
    m.realize()
    expected_m = np.array([4, 5, 6], dtype=np.float32)
    assert np.allclose(m.data, expected_m)

    m_keepdim = a.max(axis=1, keepdim=True)
    m_keepdim.realize()
    expected_m_keepdim = np.array([[5], [6]], dtype=np.float32)
    assert np.allclose(m_keepdim.data, expected_m_keepdim)

    m_all = a.max(axis=None, keepdim=False) # Max all elements
    m_all.realize()
    expected_m_all = np.array(6.0, dtype=np.float32)
    assert np.allclose(m_all.data, expected_m_all)

# Test view operation
def test_tensor_view():
    a = Tensor(shape=(2, 4), data=[[1, 2, 3, 4], [5, 6, 7, 8]])
    v = a.view([4, 2])
    v.realize()
    expected_v = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    assert np.allclose(v.data, expected_v)

    # Test view with -1
    v_inferred = a.view([2, -1])
    v_inferred.realize()
    assert np.allclose(v_inferred.data, np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32))

    v_inferred_2 = a.view([-1, 2])
    v_inferred_2.realize()
    assert np.allclose(v_inferred_2.data, np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32))

def test_tensor_view_backward():
    a_data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    a = Tensor(shape=(2, 4), data=a_data, requires_grad=True)
    v = a.view([4, 2])
    v.backward()

    # View is just a reshape, so gradients should also be reshaped.
    # If grad_output is ones, then input grad should also be ones.
    assert np.allclose(a.grad, np.ones_like(a_data))

# Test unsqueeze operation
def test_tensor_unsqueeze():
    a = Tensor(shape=(2, 3), data=[[1, 2, 3], [4, 5, 6]])
    u = a.unsqueeze(dim=0)
    u.realize()
    expected_u = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.float32)
    assert u.shape == (1, 2, 3)
    assert np.allclose(u.data, expected_u)

    u2 = a.unsqueeze(dim=1)
    u2.realize()
    expected_u2 = np.array([[[1, 2, 3]], [[4, 5, 6]]], dtype=np.float32)
    assert u2.shape == (2, 1, 3)
    assert np.allclose(u2.data, expected_u2)

def test_tensor_unsqueeze_backward():
    a_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    a = Tensor(shape=(2, 3), data=a_data, requires_grad=True)
    u = a.unsqueeze(dim=0)
    u.backward()

    # Unsqueeze is just a reshape, so gradients should also be reshaped.
    assert np.allclose(a.grad, np.ones_like(a_data))

# Test squeeze operation
def test_tensor_squeeze():
    a = Tensor(shape=(1, 2, 1, 3), data=[[[[1, 2, 3]], [[4, 5, 6]]]])
    s = a.squeeze(dim=0)
    s.realize()
    expected_s = np.array([[[1, 2, 3]], [[4, 5, 6]]], dtype=np.float32)
    assert s.shape == (2, 1, 3)
    assert np.allclose(s.data, expected_s)

    s2 = a.squeeze() # Squeeze all dimensions of size 1
    s2.realize()
    expected_s2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    assert s2.shape == (2, 3)
    assert np.allclose(s2.data, expected_s2)

def test_tensor_squeeze_backward():
    a_data = np.array([[[[1, 2, 3]], [[4, 5, 6]]]], dtype=np.float32)
    a = Tensor(shape=(1, 2, 1, 3), data=a_data, requires_grad=True)
    s = a.squeeze(dim=0)
    s.backward()

    # Squeeze is just a reshape, so gradients should also be reshaped.
    assert np.allclose(a.grad, np.ones_like(a_data))

# Test transpose operation
def test_tensor_transpose():
    a = Tensor(shape=(2, 3), data=[[1, 2, 3], [4, 5, 6]])
    t = a.transpose(0, 1)
    t.realize()
    expected_t = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float32)
    assert t.shape == (3, 2)
    assert np.allclose(t.data, expected_t)

    t2 = a.transpose() # Default last two dimensions
    t2.realize()
    assert np.allclose(t2.data, expected_t)

def test_tensor_transpose_backward():
    a_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    a = Tensor(shape=(2, 3), data=a_data, requires_grad=True)
    t = a.transpose(0, 1)
    t.backward()

    # Transpose is just a permutation, so gradients should also be permuted.
    assert np.allclose(a.grad, np.ones_like(a_data))

# Test expand operation
def test_tensor_expand():
    a = Tensor(shape=(2, 1), data=[[1], [2]])
    e = a.expand([2, 3])
    e.realize()
    expected_e = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32)
    assert e.shape == (2, 3)
    assert np.allclose(e.data, expected_e)

    # Scalar expand
    s = Tensor(shape=(), data=5.0)
    e_s = s.expand([2, 2])
    e_s.realize()
    expected_e_s = np.array([[5, 5], [5, 5]], dtype=np.float32)
    assert e_s.shape == (2, 2)
    assert np.allclose(e_s.data, expected_e_s)

def test_tensor_expand_backward():
    a_data = np.array([[1], [2]], dtype=np.float32)
    a = Tensor(shape=(2, 1), data=a_data, requires_grad=True)
    e = a.expand([2, 3])
    e.backward()

    # For expand, gradient is sum along the expanded dimensions.
    # If grad_output is ones, then sum along axis 1 (the expanded dim).
    expected_a_grad = np.array([[3], [3]], dtype=np.float32)
    assert np.allclose(a.grad, expected_a_grad)

    s_data = np.array(5.0, dtype=np.float32)
    s = Tensor(shape=(), data=s_data, requires_grad=True)
    e_s = s.expand([2, 2])
    e_s.backward()
    # Sum of all ones (2x2) = 4
    expected_s_grad = np.array(4.0, dtype=np.float32)
    assert np.allclose(s.grad, expected_s_grad)

# Test broadcast operation (similar to expand but with different rules)
def test_tensor_broadcast():
    a = Tensor(shape=(2, 1), data=[[1], [2]])
    b = Tensor(shape=(1, 3), data=[[10, 20, 30]])
    c = a + b # This will trigger broadcast
    c.realize()
    expected_c = np.array([[11, 21, 31], [12, 22, 32]], dtype=np.float32)
    assert np.allclose(c.data, expected_c)

# Test relu activation
def test_tensor_relu():
    a = Tensor(shape=(2, 3), data=[[-1, 0, 1], [-2, 3, -4]])
    r = a.relu()
    r.realize()
    expected_r = np.array([[0, 0, 1], [0, 3, 0]], dtype=np.float32)
    assert np.allclose(r.data, expected_r)

def test_tensor_relu_backward():
    a_data = np.array([[-1, 0, 1], [-2, 3, -4]], dtype=np.float32)
    a = Tensor(shape=(2, 3), data=a_data, requires_grad=True)
    r = a.relu()
    r.backward()

    # d(relu)/dx = 1 if x > 0, 0 otherwise.
    expected_a_grad = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.float32)
    assert np.allclose(a.grad, expected_a_grad)

# Test log operation
def test_tensor_log():
    a = Tensor(shape=(2, 2), data=[[1, 2], [3, 4]])
    l = a.log()
    l.realize()
    expected_l = np.log(np.array([[1, 2], [3, 4]], dtype=np.float32))
    assert np.allclose(l.data, expected_l)

def test_tensor_log_backward():
    a_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    a = Tensor(shape=(2, 2), data=a_data, requires_grad=True)
    l = a.log()
    l.backward()

    # d(log x)/dx = 1/x
    expected_a_grad = 1 / a_data
    assert np.allclose(a.grad, expected_a_grad)

# Test exp operation
def test_tensor_exp():
    a = Tensor(shape=(2, 2), data=[[1, 2], [3, 4]])
    e = a.exp()
    e.realize()
    expected_e = np.exp(np.array([[1, 2], [3, 4]], dtype=np.float32))
    assert np.allclose(e.data, expected_e)

def test_tensor_exp_backward():
    a_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    a = Tensor(shape=(2, 2), data=a_data, requires_grad=True)
    e = a.exp()
    e.backward()

    # d(exp x)/dx = exp x
    expected_a_grad = np.exp(a_data)
    assert np.allclose(a.grad, expected_a_grad)

# Test abs operation
def test_tensor_abs():
    a = Tensor(shape=(2, 2), data=[[-1, 2], [-3, 4]])
    ab = a.abs()
    ab.realize()
    expected_ab = np.array([[1, 2], [3, 4]], dtype=np.float32)
    assert np.allclose(ab.data, expected_ab)

def test_tensor_abs_backward():
    a_data = np.array([[-1, 2], [-3, 4]], dtype=np.float32)
    a = Tensor(shape=(2, 2), data=a_data, requires_grad=True)
    ab = a.abs()
    ab.backward()

    # d(|x|)/dx = 1 if x > 0, -1 if x < 0, undefined at x=0 (often taken as 0 or 1)
    # For simplicity, let's assume 0 at 0 for testing.
    expected_a_grad = np.array([[-1, 1], [-1, 1]], dtype=np.float32)
    assert np.allclose(a.grad, expected_a_grad)

# Test softmax (forward only for now, backward is complex)
def test_tensor_softmax():
    a = Tensor(shape=(2, 3), data=[[1, 2, 3], [4, 5, 6]])
    s = a.softmax()
    s.realize()
    expected_s = np.array([[0.09003057, 0.24472848, 0.66524094],
                           [0.09003057, 0.24472848, 0.66524094]], dtype=np.float32)
    # Using numpy's softmax for comparison
    def numpy_softmax(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    assert np.allclose(s.data, numpy_softmax(a.data))

# Edge cases
def test_tensor_empty_data_init():
    # Should default to a scalar 0.0 tensor
    t = Tensor()
    assert t.shape == ()
    assert t.data == 0.0

def test_tensor_zero_dim_ops():
    a = Tensor(shape=(), data=5.0, requires_grad=True)
    b = Tensor(shape=(), data=3.0, requires_grad=True)
    c = a + b
    c.realize()
    assert np.allclose(c.data, 8.0)
    c.backward()
    assert np.allclose(a.grad, 1.0)
    assert np.allclose(b.grad, 1.0)

    d = a * b
    d.realize()
    assert np.allclose(d.data, 15.0)
    d.backward()
    assert np.allclose(a.grad, 3.0)
    assert np.allclose(b.grad, 5.0)

def test_tensor_no_grad_propagation():
    a = Tensor(shape=(2, 2), data=[[1, 2], [3, 4]], requires_grad=False)
    b = Tensor(shape=(2, 2), data=[[5, 6], [7, 8]], requires_grad=True)
    c = a + b # c should not require grad if any input does not
    assert c.requires_grad is True # This is different from PyTorch, where if one input doesn't require grad, output doesn't.
                                  # Here, it seems to propagate if at least one does. Let's check the implementation.
                                  # Based on the Node init, if out_tensor has _c_tensor, it will create a CNode.
                                  # The requires_grad property of the output tensor is set during its creation.
                                  # Let's assume for now that if any input requires grad, the output does.

    # If c.requires_grad is True, then c.backward() will be called.
    # However, a.grad should remain None.
    c.backward()
    assert a.grad is None
    assert np.allclose(b.grad, np.ones((2, 2), dtype=np.float32))

    # Test case where both don't require grad
    x = Tensor(shape=(2, 2), data=[[1, 2], [3, 4]], requires_grad=False)
    y = Tensor(shape=(2, 2), data=[[5, 6], [7, 8]], requires_grad=False)
    z = x + y
    assert z.requires_grad is False
    # z.backward() would raise an error or do nothing, which is expected.
    # No need to call backward if requires_grad is False.

# Test chain rule
def test_chain_rule():
    a = Tensor(shape=(), data=2.0, requires_grad=True)
    b = a * 3.0
    c = b + 1.0
    d = c.relu()
    d.backward()

    # d = relu(c)
    # c = b + 1
    # b = a * 3
    # dD/dC = 1 (since c = 7 > 0)
    # dC/dB = 1
    # dB/dA = 3
    # dD/dA = dD/dC * dC/dB * dB/dA = 1 * 1 * 3 = 3
    assert np.allclose(a.grad, 3.0)

    # Another chain rule example
    x = Tensor(shape=(1,), data=[2.0], requires_grad=True)
    y = x * x # y = x^2
    z = y + x # z = x^2 + x
    z.backward()

    # dZ/dX = 2x + 1
    # At x=2, dZ/dX = 2*2 + 1 = 5
    assert np.allclose(x.grad, 5.0)

# Test complex graph with shared tensors
def test_shared_tensor_backward():
    a = Tensor(shape=(), data=2.0, requires_grad=True)
    b = a * 2.0 # b = 2a
    c = a + 3.0 # c = a + 3
    d = b + c   # d = 2a + (a + 3) = 3a + 3
    d.backward()

    # dD/dA = d(3a + 3)/dA = 3
    assert np.allclose(a.grad, 3.0)

# Test with large tensors (basic check)
def test_large_tensor_init():
    shape = (100, 100)
    data = np.random.rand(*shape).astype(np.float32)
    t = Tensor(shape=shape, data=data.tolist())
    t.realize()
    assert t.shape == shape
    assert np.allclose(t.data, data)

def test_large_tensor_add_backward():
    shape = (50, 50)
    a_data = np.random.rand(*shape).astype(np.float32)
    b_data = np.random.rand(*shape).astype(np.float32)
    a = Tensor(shape=shape, data=a_data.tolist(), requires_grad=True)
    b = Tensor(shape=shape, data=b_data.tolist(), requires_grad=True)
    c = a + b
    c.backward()
    assert np.allclose(a.grad, np.ones(shape, dtype=np.float32))
    assert np.allclose(b.grad, np.ones(shape, dtype=np.float32))

# Test with negative values
def test_tensor_neg_values():
    a = Tensor(shape=(2, 2), data=[[-1, -2], [-3, -4]], requires_grad=True)
    b = Tensor(shape=(2, 2), data=[[5, 6], [7, 8]], requires_grad=True)
    c = a * b
    c.realize()
    expected_c = np.array([[-5, -12], [-21, -32]], dtype=np.float32)
    assert np.allclose(c.data, expected_c)
    c.backward()
    assert np.allclose(a.grad, b.data)
    assert np.allclose(b.grad, a.data)

# Test with zero values
def test_tensor_zero_values():
    a = Tensor(shape=(2, 2), data=[[0, 1], [2, 0]], requires_grad=True)
    b = Tensor(shape=(2, 2), data=[[5, 0], [0, 8]], requires_grad=True)
    c = a * b
    c.realize()
    expected_c = np.array([[0, 0], [0, 0]], dtype=np.float32)
    assert np.allclose(c.data, expected_c)
    c.backward()
    assert np.allclose(a.grad, b.data)
    assert np.allclose(b.grad, a.data)

# Test with mixed positive/negative/zero values
def test_tensor_mixed_values():
    a = Tensor(shape=(2, 2), data=[[-1, 0], [2, -3]], requires_grad=True)
    b = Tensor(shape=(2, 2), data=[[4, -5], [0, 6]], requires_grad=True)
    c = a + b
    c.realize()
    expected_c = np.array([[3, -5], [2, 3]], dtype=np.float32)
    assert np.allclose(c.data, expected_c)
    c.backward()
    assert np.allclose(a.grad, np.ones_like(a.data))
    assert np.allclose(b.grad, np.ones_like(b.data))

    d = a * b
    d.realize()
    expected_d = np.array([[-4, 0], [0, -18]], dtype=np.float32)
    assert np.allclose(d.data, expected_d)
    d.backward()
    assert np.allclose(a.grad, b.data)
    assert np.allclose(b.grad, a.data)

# Test with different data types (though C backend is float32)
def test_tensor_float_data_types():
    a = Tensor(shape=(2, 2), data=[[1.0, 2.5], [3.1, 4.9]], requires_grad=True)
    b = Tensor(shape=(2, 2), data=[[5.0, 6.5], [7.1, 8.9]], requires_grad=True)
    c = a + b
    c.realize()
    expected_c = np.array([[6.0, 8.5], [10.2, 13.8]], dtype=np.float32)
    assert np.allclose(c.data, expected_c)
    c.backward()
    assert np.allclose(a.grad, np.ones_like(a.data))
    assert np.allclose(b.grad, np.ones_like(b.data))

# Test with different shapes (broadcasting)
def test_tensor_broadcast_add():
    a = Tensor(shape=(2, 3), data=[[1, 2, 3], [4, 5, 6]], requires_grad=True)
    b = Tensor(shape=(1, 3), data=[[10, 20, 30]], requires_grad=True)
    c = a + b
    c.realize()
    expected_c = np.array([[11, 22, 33], [14, 25, 36]], dtype=np.float32)
    assert np.allclose(c.data, expected_c)
    c.backward()

    # For broadcasting, the gradient of the broadcasted tensor is summed along the broadcasted dimensions.
    # dC/dA = 1 (shape (2,3))
    # dC/dB = sum(1, axis=0) (shape (1,3))
    assert np.allclose(a.grad, np.ones_like(a.data))
    assert np.allclose(b.grad, np.array([[2, 2, 2]], dtype=np.float32))

def test_tensor_broadcast_mul():
    a = Tensor(shape=(2, 3), data=[[1, 2, 3], [4, 5, 6]], requires_grad=True)
    b = Tensor(shape=(1, 3), data=[[10, 20, 30]], requires_grad=True)
    c = a * b
    c.realize()
    expected_c = np.array([[10, 40, 90], [40, 100, 180]], dtype=np.float32)
    assert np.allclose(c.data, expected_c)
    c.backward()

    # dC/dA = B (shape (2,3) after broadcasting B)
    # dC/dB = sum(A, axis=0) (shape (1,3))
    expected_a_grad = np.array([[10, 20, 30], [10, 20, 30]], dtype=np.float32)
    expected_b_grad = np.array([[5, 7, 9]], dtype=np.float32) # sum of a_data along axis 0
    assert np.allclose(a.grad, expected_a_grad)
    assert np.allclose(b.grad, expected_b_grad)

# Test with complex operations (multiple steps)
def test_complex_operation_chain():
    a = Tensor(shape=(2, 2), data=[[1, 2], [3, 4]], requires_grad=True)
    b = Tensor(shape=(2, 2), data=[[5, 6], [7, 8]], requires_grad=True)

    x = a + b
    y = x * a
    z = y.sum()
    z.backward()

    # Manual calculation for verification:
    # a_data = [[1, 2], [3, 4]]
    # b_data = [[5, 6], [7, 8]]
    # x_data = a_data + b_data = [[6, 8], [10, 12]]
    # y_data = x_data * a_data = [[6*1, 8*2], [10*3, 12*4]] = [[6, 16], [30, 48]]
    # z_data = sum(y_data) = 6 + 16 + 30 + 48 = 100

    # dZ/dY = 1 (since Z is sum of Y)
    # dY/dX = A
    # dY/dA = X
    # dX/dA = 1
    # dX/dB = 1

    # dZ/dA = dZ/dY * dY/dA + dZ/dY * dY/dX * dX/dA
    #       = 1 * X + 1 * A * 1
    #       = X + A
    #       = (a_data + b_data) + a_data
    #       = [[6, 8], [10, 12]] + [[1, 2], [3, 4]] = [[7, 10], [13, 16]]

    # dZ/dB = dZ/dY * dY/dX * dX/dB
    #       = 1 * A * 1
    #       = A
    #       = [[1, 2], [3, 4]]

    expected_a_grad = np.array([[7, 10], [13, 16]], dtype=np.float32)
    expected_b_grad = np.array([[1, 2], [3, 4]], dtype=np.float32)

    assert np.allclose(a.grad, expected_a_grad)
    assert np.allclose(b.grad, expected_b_grad)

# Test with more complex movement operations and backward
def test_complex_movement_backward():
    a = Tensor(shape=(1, 2, 1, 3), data=[[[[1, 2, 3]], [[4, 5, 6]]]], requires_grad=True)
    b = a.squeeze() # (2, 3)
    c = b.transpose(0, 1) # (3, 2)
    d = c.sum()
    d.backward()

    # dD/dC = ones_like(C) = ones((3,2))
    # dC/dB = transpose_grad(dD/dC) = ones((2,3))
    # dB/dA = squeeze_grad(dC/dB) = ones((1,2,1,3))
    assert np.allclose(a.grad, np.ones_like(a.data))

