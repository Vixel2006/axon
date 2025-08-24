import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import nawah_api as nw

DEVICES = ["cpu", "cuda:0"]


def skip_if_cuda_not_available(device):
    """A pytest helper to skip tests if the required 'cuda' device is not available."""
    if device == "cuda" and "cuda" not in DEVICES:
        pytest.skip("Skipping test: CUDA device not available or configured.")


class TestTensorOps:
    """
    Test suite for nawah.Tensor operations using pytest.
    """

    @pytest.mark.parametrize("device", DEVICES)
    def test_creation_and_properties(self, device):
        """Test basic tensor creation and property access on all available devices."""
        skip_if_cuda_not_available(device)
        data = [[1.0, 2.0], [3.0, 4.0]]
        t = nw.Tensor.from_data(data, device=device)

        assert t.shape == [2, 2]
        assert t.device.type == (
            nw.DeviceType.CUDA if device == "cuda:0" else nw.DeviceType.CPU
        )
        assert t.dtype == nw.DType.float32
        # .data property fetches data from the device to the host (as numpy array)
        assert np.allclose(t.data, data)

    @pytest.mark.parametrize("device", DEVICES)
    def test_addition(self, device):
        """Test element-wise tensor addition."""
        skip_if_cuda_not_available(device)
        a = nw.Tensor([[1, 2], [3, 4]], device=device)
        b = nw.Tensor([[5, 6], [7, 8]], device=device)

        c = a + b

        expected = np.array([[1, 2], [3, 4]]) + np.array([[5, 6], [7, 8]])
        assert c.shape == list(expected.shape)
        assert np.allclose(c.data, expected)

    @pytest.mark.parametrize("device", DEVICES)
    def test_subtraction(self, device):
        """Test element-wise tensor subtraction."""
        skip_if_cuda_not_available(device)
        a = nw.Tensor([[10, 20], [30, 40]], device=device)
        b = nw.Tensor([[1, 2], [3, 4]], device=device)

        c = a - b

        expected = np.array([[10, 20], [30, 40]]) - np.array([[1, 2], [3, 4]])
        assert c.shape == list(expected.shape)
        assert np.allclose(c.data, expected)

    @pytest.mark.parametrize("device", DEVICES)
    def test_elementwise_multiplication(self, device):
        """Test element-wise tensor multiplication."""
        skip_if_cuda_not_available(device)
        a = nw.Tensor([[1, 2], [3, 4]], device=device)
        b = nw.Tensor([[5, 6], [7, 8]], device=device)

        c = a * b

        expected = np.array([[1, 2], [3, 4]]) * np.array([[5, 6], [7, 8]])
        assert c.shape == list(expected.shape)
        assert np.allclose(c.data, expected)

    @pytest.mark.parametrize("device", DEVICES)
    def test_tensor_division(self, device):
        """Test element-wise tensor division (tensor / tensor)."""
        skip_if_cuda_not_available(device)
        a = nw.Tensor([[8, 18], [30, 40]], device=device)
        b = nw.Tensor([[2, 3], [3, 5]], device=device)

        c = a / b

        expected = np.array([[8, 18], [30, 40]]) / np.array([[2, 3], [3, 5]])
        assert c.shape == list(expected.shape)
        assert np.allclose(c.data, expected)

    @pytest.mark.parametrize("device", DEVICES)
    def test_scalar_division(self, device):
        """Test element-wise tensor division by a scalar (tensor / scalar)."""
        skip_if_cuda_not_available(device)
        a = nw.Tensor([[10, 20], [30, 40]], device=device)
        scalar = 2.0

        c = a / scalar

        expected = np.array([[10, 20], [30, 40]]) / scalar
        assert c.shape == list(expected.shape)
        assert np.allclose(c.data, expected)

    @pytest.mark.parametrize("device", DEVICES)
    def test_matrix_multiplication(self, device):
        """Test matrix multiplication."""
        skip_if_cuda_not_available(device)
        a = nw.Tensor([[1, 2, 3], [4, 5, 6]], device=device)
        b = nw.Tensor([[7, 8], [9, 10], [11, 12]], device=device)

        c = a @ b

        np_a = np.array([[1, 2, 3], [4, 5, 6]])
        np_b = np.array([[7, 8], [9, 10], [11, 12]])
        expected = np_a @ np_b

        assert c.shape == list(expected.shape)
        assert np.allclose(c.data, expected)

    @pytest.mark.parametrize("device", DEVICES)
    def test_exp(self, device):
        """Test element-wise exponential function."""
        skip_if_cuda_not_available(device)
        a = nw.Tensor([[1, -2], [0, 3]], device=device)
        c = nw.exp(a)
        expected = np.exp(np.array([[1, -2], [0, 3]]))
        assert c.shape == list(expected.shape)
        assert np.allclose(c.data, expected)

    @pytest.mark.parametrize("device", DEVICES)
    def test_log(self, device):
        """Test element-wise natural logarithm function."""
        skip_if_cuda_not_available(device)
        # Log requires positive values
        a = nw.Tensor([[1, 2], [10, 20]], device=device)
        c = nw.log(a)
        expected = np.log(np.array([[1, 2], [10, 20]]))
        assert c.shape == list(expected.shape)
        assert np.allclose(c.data, expected)

    @pytest.mark.parametrize("device", DEVICES)
    def test_softmax(self, device):
        """Test row-wise softmax function."""
        skip_if_cuda_not_available(device)
        a = nw.Tensor([[1, 2, 3], [-1, 0, 1]], device=device)

        # Perform softmax with your library
        c = a >> nw.softmax

        # Calculate expected result with numpy using a numerically stable method
        np_a = np.array([[1, 2, 3], [-1, 0, 1]], dtype=np.float32)
        e_x = np.exp(np_a - np.max(np_a, axis=1, keepdims=True))
        expected = e_x / np.sum(e_x, axis=1, keepdims=True)

        assert c.shape == list(expected.shape)
        assert np.allclose(c.data, expected, atol=1e-6)

    @pytest.mark.parametrize("device", DEVICES)
    def test_autograd_context_creation(self, device):
        """Verify that the Tape context is correctly created after operations."""
        skip_if_cuda_not_available(device)
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

        # Test division
        e = a / b
        assert e.requires_grad is True
        assert e.ctx is not None

    def test_sum_reductions_cpu(self):
        """Test sum reduction on CPU."""
        t = nw.Tensor([[1, 2, 3], [4, 5, 6]], device="cpu")
        np_t = np.array([[1, 2, 3], [4, 5, 6]])

        s2 = t.sum(dim=1)
        assert s2.shape == [2]
        assert np.allclose(s2.data, np.sum(np_t, axis=1))

        s3 = t.sum(dim=0, keepdim=True)
        assert s3.shape == [1, 3]
        assert np.allclose(s3.data, np.sum(np_t, axis=0, keepdims=True))

    def test_mean_reductions_cpu(self):
        """Test mean reduction on CPU."""
        t = nw.Tensor([[1, 2, 3], [4, 5, 6]], device="cpu", dtype=nw.DType.float32)
        np_t = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

        m2 = t.mean(dim=0)
        assert m2.shape == [3]
        assert np.allclose(m2.data, np.mean(np_t, axis=0))

        m3 = t.mean(dim=1, keepdim=True)
        assert m3.shape == [2, 1]
        assert np.allclose(m3.data, np.mean(np_t, axis=1, keepdims=True))
