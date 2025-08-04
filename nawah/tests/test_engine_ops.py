import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import nawah_api as nw


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

    @pytest.mark.parametrize("device", DEVICES)
    def test_creation_and_properties(self, device):
        """Test basic tensor creation and property access."""
        skip_if_cuda_not_available(device)
        data = [[1.0, 2.0], [3.0, 4.0]]
        t = nw.Tensor(data, device=device)

        assert t.shape == [2, 2]
        assert t.device.type == (nw.DeviceType.CUDA if device == "cuda" else nw.DeviceType.CPU)
        assert t.dtype == nw.DType.float32
        assert np.allclose(t.data, data)

    @pytest.mark.parametrize("device", DEVICES)
    def test_addition(self, device):
        """Test element-wise tensor addition."""
        skip_if_cuda_not_available(device)
        a = nw.Tensor([[1, 2], [3, 4]], device=device)
        b = nw.Tensor([[5, 6], [7, 8]], device=device)
        
        # Perform operation with your library
        c = a + b
        
        # Verify with numpy
        expected = np.array([[1, 2], [3, 4]]) + np.array([[5, 6], [7, 8]])
        assert c.shape == [2, 2]
        assert np.allclose(c.data, expected)

    @pytest.mark.parametrize("device", DEVICES)
    def test_subtraction(self, device):
        """Test element-wise tensor subtraction."""
        skip_if_cuda_not_available(device)
        a = nw.Tensor([[10, 20], [30, 40]], device=device)
        b = nw.Tensor([[1, 2], [3, 4]], device=device)
        
        c = a - b
        
        expected = np.array([[10, 20], [30, 40]]) - np.array([[1, 2], [3, 4]])
        assert c.shape == [2, 2]
        assert np.allclose(c.data, expected)

    @pytest.mark.parametrize("device", DEVICES)
    def test_elementwise_multiplication(self, device):
        """Test element-wise tensor multiplication."""
        skip_if_cuda_not_available(device)
        a = nw.Tensor([[1, 2], [3, 4]], device=device)
        b = nw.Tensor([[5, 6], [7, 8]], device=device)
        
        c = a * b
        
        expected = np.array([[1, 2], [3, 4]]) * np.array([[5, 6], [7, 8]])
        assert c.shape == [2, 2]
        assert np.allclose(c.data, expected)

    @pytest.mark.parametrize("device", DEVICES)
    def test_matrix_multiplication(self, device):
        """Test matrix multiplication."""
        skip_if_cuda_not_available(device)
        a = nw.Tensor([[1, 2, 3], [4, 5, 6]], device=device)       # Shape (2, 3)
        b = nw.Tensor([[7, 8], [9, 10], [11, 12]], device=device) # Shape (3, 2)
        
        c = a @ b
        
        np_a = np.array([[1, 2, 3], [4, 5, 6]])
        np_b = np.array([[7, 8], [9, 10], [11, 12]])
        expected = np_a @ np_b
        
        assert c.shape == [2, 2]
        assert np.allclose(c.data, expected)

    def test_sum_reductions(self):
        """Test sum reduction on CPU."""
        # Reductions are only implemented for CPU in the prompt
        t = nw.Tensor([[1, 2, 3], [4, 5, 6]], device="cpu")
        np_t = np.array([[1, 2, 3], [4, 5, 6]])


        # Case 2: Reduce along a dimension
        s2 = t.sum(dim=1)
        assert s2.shape == [2]
        assert np.allclose(s2.data, np.sum(np_t, axis=1))
        
        # Case 3: Reduce along a dimension and keepdim
        s3 = t.sum(dim=0, keepdim=True)
        assert s3.shape == [1, 3]
        assert np.allclose(s3.data, np.sum(np_t, axis=0, keepdims=True))

    def test_mean_reductions(self):
        """Test mean reduction on CPU."""
        # Reductions are only implemented for CPU in the prompt
        t = nw.Tensor([[1, 2, 3], [4, 5, 6]], device="cpu", dtype=nw.DType.float32)
        np_t = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)


        # Case 2: Reduce along a dimension
        m2 = t.mean(dim=0)
        assert m2.shape == [3]
        assert np.allclose(m2.data, np.mean(np_t, axis=0))

        # Case 3: Reduce along a dimension and keepdim
        m3 = t.mean(dim=1, keepdim=True)
        assert m3.shape == [2, 1]
        assert np.allclose(m3.data, np.mean(np_t, axis=1, keepdims=True))

    @pytest.mark.parametrize("device", DEVICES)
    def test_autograd_context_creation(self, device):
        """Verify that the Tape context is correctly created after an operation."""
        skip_if_cuda_not_available(device)
        # One tensor must require grad for the output to require grad
        a = nw.Tensor([[1, 2]], device=device, requires_grad=True)
        b = nw.Tensor([[3, 4]], device=device, requires_grad=False)
        
        # Test addition
        c = a + b
        assert c.requires_grad is True
        assert c.ctx is not None
        assert len(c.ctx.prev) == 2

        # Test multiplication
        d = a * b
        assert d.requires_grad is True
        assert d.ctx is not None
