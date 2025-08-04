import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import nawah_api as nw

#
# ... (Keep all your existing code from the top of the file) ...
#

def is_cuda_available():
    """Check if CUDA is available and a tensor can be created on the GPU."""
    try:
        nw.Tensor([1], device="cuda")
        nw.cuda_synchronize()
        return True
    except (RuntimeError, ValueError) as e:
        print(f"CUDA not available: {e}")
        return False

DEVICES = ["cpu"]
if is_cuda_available():
    DEVICES.append("cuda")

def skip_if_cuda_not_available(device):
    if device == "cuda" and "cuda" not in DEVICES:
        pytest.skip("CUDA device not available")


class TestTensorOps:
    """
    Test suite for nawah.Tensor operations using pytest.
    """

    # ... (Keep your existing forward-pass tests: test_creation_and_properties, test_addition, etc.) ...
    
    # --- Autograd / Backward Pass Tests ---
    # The following tests verify the correctness of the backward() pass.

    @pytest.mark.parametrize("device", DEVICES)
    def test_backward_addition(self, device):
        """Tests the backward pass for a simple addition."""
        skip_if_cuda_not_available(device)
        a = nw.Tensor([[1, 2, 3]], requires_grad=True, device=device)
        b = nw.Tensor([[4, 5, 6]], requires_grad=True, device=device)
        
        # The backward pass is started from a scalar, so we sum the result.
        c = (a + b)
        c.backward()

        # The derivative of sum is 1. For `c = a + b`, the gradient of `a` is 1 
        # and the gradient of `b` is 1, multiplied by the output gradient.
        # So both grads should be tensors of 1s.
        expected_grad = np.array([[1., 1., 1.]])
        assert np.allclose(a.grad, expected_grad)
        assert np.allclose(b.grad, expected_grad)

    @pytest.mark.parametrize("device", DEVICES)
    def test_backward_subtraction(self, device):
        """Tests the backward pass for a simple subtraction."""
        skip_if_cuda_not_available(device)
        a = nw.Tensor([[10, 20, 30]], requires_grad=True, device=device)
        b = nw.Tensor([[1, 2, 3]], requires_grad=True, device=device)
        
        c = (a - b)
        c.backward()

        # For `c = a - b`, the gradient of `a` is 1, but the gradient of `b` is -1.
        expected_grad_a = np.array([[1., 1., 1.]])
        expected_grad_b = np.array([[-1., -1., -1.]])
        assert np.allclose(a.grad, expected_grad_a)
        assert np.allclose(b.grad, expected_grad_b)

    @pytest.mark.parametrize("device", DEVICES)
    def test_backward_multiplication(self, device):
        """Tests the backward pass for element-wise multiplication."""
        skip_if_cuda_not_available(device)
        # Use simple data to easily verify the chain rule
        a_data = [[2., 5., 10.]]
        b_data = [[3., 4., 6.]]
        a = nw.Tensor(a_data, requires_grad=True, device=device)
        b = nw.Tensor(b_data, requires_grad=True, device=device)
        
        c = (a * b)
        c.backward()

        # For `c = a * b`, the gradient of `a` is `b`'s data,
        # and the gradient of `b` is `a`'s data.
        assert np.allclose(a.grad, b_data) # grad(a) == b
        assert np.allclose(b.grad, a_data) # grad(b) == a

    @pytest.mark.parametrize("device", DEVICES)
    def test_backward_chain_rule_mul_add(self, device):
        """
        Tests the backward pass for a chain of operations (mul -> add).
        This is the exact test case you provided.
        """
        skip_if_cuda_not_available(device)
        n1_data = [[1., 3., 4.]]
        n2_data = [[3., 4., 5.]]
        n4_data = [[4., 5., 6.]]
        
        n1 = nw.Tensor(n1_data, requires_grad=True, device=device)
        n2 = nw.Tensor(n2_data, requires_grad=True, device=device)
        n4 = nw.Tensor(n4_data, requires_grad=True, device=device)

        # The graph:
        n3 = n1 * n2
        n5 = n3 + n4
        
        # Start backward pass from a scalar sum
        loss = n5
        loss.backward()
        
        # Verify gradients step-by-step
        # 1. Gradient of loss w.r.t n5 is 1.
        # 2. Grad of n4 is 1. Grad of n3 is 1.
        assert np.allclose(n4.grad, [[1., 1., 1.]])
        # 3. Grad of n1 is grad_n3 * n2_data = 1 * n2_data
        assert np.allclose(n1.grad, n2_data)
        # 4. Grad of n2 is grad_n3 * n1_data = 1 * n1_data
        assert np.allclose(n2.grad, n1_data)

