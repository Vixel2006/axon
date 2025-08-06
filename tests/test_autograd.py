import pytest
import numpy as np
import sys
import os

# Ensure the library is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import nawah_api as nw


DEVICES = ["cpu", "cuda:0"]

def skip_if_cuda_not_available(device):
    """A pytest helper to skip tests if the required 'cuda' device is not available."""
    if device == "cuda:0" and "cuda:0" not in DEVICES:
        pytest.skip("Skipping test: CUDA device not available or configured.")


class TestTensorAutograd:
    """
    Test suite for the backward pass and gradient calculations in nawah.
    """

    @pytest.mark.parametrize("device", DEVICES)
    def test_backward_addition(self, device):
        """Tests the backward pass for a simple addition."""
        skip_if_cuda_not_available(device)
        a = nw.Tensor([[1, 2, 3]], requires_grad=True, device=device)
        b = nw.Tensor([[4, 5, 6]], requires_grad=True, device=device)
        
        # We the result to get a scalar loss, which starts the backward pass with a gradient of 1.
        c = a + b
        c.backward()

        # The derivative of is 1. For `L = sum(a + b)`, dL/da = 1 and dL/db = 1.
        expected_grad = np.array([[1., 1., 1.]])
        assert a.grad is not None and np.allclose(a.grad, expected_grad)
        assert b.grad is not None and np.allclose(b.grad, expected_grad)

    @pytest.mark.parametrize("device", DEVICES)
    def test_backward_subtraction(self, device):
        """Tests the backward pass for a simple subtraction."""
        skip_if_cuda_not_available(device)
        a = nw.Tensor([[10, 20, 30]], requires_grad=True, device=device)
        b = nw.Tensor([[1, 2, 3]], requires_grad=True, device=device)
        
        c = a - b
        c.backward()

        # For `L =(a - b)`, dL/da = 1, but dL/db = -1.
        expected_grad_a = np.array([[1., 1., 1.]])
        expected_grad_b = np.array([[-1., -1., -1.]])
        assert a.grad is not None and np.allclose(a.grad, expected_grad_a)
        assert b.grad is not None and np.allclose(b.grad, expected_grad_b)

    @pytest.mark.parametrize("device", DEVICES)
    def test_backward_multiplication(self, device):
        """Tests the backward pass for element-wise multiplication."""
        skip_if_cuda_not_available(device)
        a_data = [[2., 5., 10.]]
        b_data = [[3., 4., 6.]]
        a = nw.Tensor(a_data, requires_grad=True, device=device)
        b = nw.Tensor(b_data, requires_grad=True, device=device)
        
        c = a * b
        c.backward()

        # For `L =(a * b)`, dL/da = b and dL/db = a.
        assert a.grad is not None and np.allclose(a.grad, b_data)
        assert b.grad is not None and np.allclose(b.grad, a_data)

    @pytest.mark.parametrize("device", DEVICES)
    def test_backward_division(self, device):
        """Tests the backward pass for element-wise division."""
        skip_if_cuda_not_available(device)
        a_data = np.array([[8., 18., 40.]])
        b_data = np.array([[2., 3., 5.]])
        a = nw.Tensor([[8., 18., 40.]], requires_grad=True, device=device)
        b = nw.Tensor([[2., 3., 5.]], requires_grad=True, device=device)

        c = a / b
        c.backward()

        # For L =(a/b), dL/da = 1/b and dL/db = -a / (b**2)
        expected_grad_a = 1.0 / b_data
        expected_grad_b = -a_data / (b_data ** 2)
        assert a.grad is not None and np.allclose(a.grad, expected_grad_a)
        assert b.grad is not None and np.allclose(b.grad, expected_grad_b)
        
    @pytest.mark.parametrize("device", DEVICES)
    def test_backward_scalar_division(self, device):
        """Tests the backward pass for division by a scalar."""
        skip_if_cuda_not_available(device)
        a_data = np.array([[10., 20., 30.]])
        scalar = 2.0
        a = nw.Tensor([[10., 20., 30.]], requires_grad=True, device=device)

        c = a / scalar
        c.backward()

        # For L =(a/k), dL/da = 1/k
        expected_grad_a = np.full(a_data.shape, 1.0 / scalar)
        assert a.grad is not None and np.allclose(a.grad, expected_grad_a)

    @pytest.mark.parametrize("device", DEVICES)
    def test_backward_exp(self, device):
        """Tests the backward pass for the exp function."""
        skip_if_cuda_not_available(device)
        a_data = np.array([[1., -2., 0.]])
        a = nw.Tensor([[1., -2., 0.]], requires_grad=True, device=device)
        
        c = nw.exp(a)
        c.backward()

        # For L =(exp(a)), dL/da = exp(a)
        expected_grad = np.exp(a_data)
        assert a.grad is not None and np.allclose(a.grad, expected_grad)

    @pytest.mark.parametrize("device", DEVICES)
    def test_backward_log(self, device):
        """Tests the backward pass for the log function."""
        skip_if_cuda_not_available(device)
        a_data = np.array([[1., 10., 0.5]])
        a = nw.Tensor([[1., 10., 0.5]], requires_grad=True, device=device)
        
        c = nw.log(a)
        c.backward()

        # For L =(log(a)), dL/da = 1/a
        expected_grad = 1.0 / a_data
        assert a.grad is not None and np.allclose(a.grad, expected_grad)

    @pytest.mark.parametrize("device", DEVICES)
    def test_backward_matmul(self, device):
        """Tests the backward pass for matrix multiplication."""
        skip_if_cuda_not_available(device)
        a_data = np.array([[1., 2., 3.], [4., 5., 6.]]) # (2, 3)
        b_data = np.array([[7., 8.], [9., 10.], [11., 12.]]) # (3, 2)
        
        a = nw.Tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True, device=device)
        b = nw.Tensor([[7., 8.], [9., 10.], [11., 12.]], requires_grad=True, device=device)
        
        c = a @ b # Result is a scalar, c.shape is (2,2) before sum
        c.backward()

        # For L =(A @ B), dL/dA = (dL/dC) @ B.T. Since L=sum(C), dL/dC is a matrix of ones.
        grad_c = np.ones((2, 2))
        expected_grad_a = grad_c @ b_data.T
        
        # And dL/dB = A.T @ (dL/dC)
        expected_grad_b = a_data.T @ grad_c

        assert a.grad is not None and np.allclose(a.grad, expected_grad_a)
        assert b.grad is not None and np.allclose(b.grad, expected_grad_b)


