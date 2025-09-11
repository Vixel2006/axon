import pytest
import numpy as np
from idrak.core.tensor import Tensor
from idrak.functions import zeros, ones, randn, uniform, concat, stack, softmax, log_softmax, conv2d

def test_zeros():
    shape = (2, 3)
    t = zeros(shape)
    t.realize()
    assert t.shape == shape
    assert np.allclose(t.data, np.zeros(shape, dtype=np.float32))

def test_ones():
    shape = (2, 3)
    t = ones(shape)
    t.realize()
    assert t.shape == shape
    assert np.allclose(t.data, np.ones(shape, dtype=np.float32))

def test_randn():
    shape = (2, 3)
    t = randn(shape)
    t.realize()
    assert t.shape == shape
    # For random numbers, we can only check shape and type, not exact values
    assert isinstance(t.data, np.ndarray)
    assert t.data.shape == shape

def test_uniform():
    shape = (2, 3)
    low = -1.0
    high = 1.0
    t = uniform(shape, low=low, high=high)
    t.realize()
    assert t.shape == shape
    assert isinstance(t.data, np.ndarray)
    assert t.data.shape == shape
    # Check if values are within the specified range
    assert np.all(t.data >= low)
    assert np.all(t.data <= high)

def test_concat():
    a_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b_np = np.array([[5, 6], [7, 8]], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2, 2))
    b_idrak = Tensor(data=b_np, shape=(2, 2))

    # Concat along axis 0
    c_idrak_axis0 = concat([a_idrak, b_idrak], axis=0)
    c_idrak_axis0.realize()
    c_np_axis0 = np.concatenate((a_np, b_np), axis=0)
    assert c_idrak_axis0.shape == c_np_axis0.shape
    assert np.allclose(c_idrak_axis0.data, c_np_axis0)

    # Concat along axis 1
    c_idrak_axis1 = concat([a_idrak, b_idrak], axis=1)
    c_idrak_axis1.realize()
    c_np_axis1 = np.concatenate((a_np, b_np), axis=1)
    assert c_idrak_axis1.shape == c_np_axis1.shape
    assert np.allclose(c_idrak_axis1.data, c_np_axis1)

def test_stack():
    a_np = np.array([1, 2], dtype=np.float32)
    b_np = np.array([3, 4], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2,))
    b_idrak = Tensor(data=b_np, shape=(2,))

    # Stack along axis 0
    c_idrak_axis0 = stack([a_idrak, b_idrak], axis=0)
    c_idrak_axis0.realize()
    c_np_axis0 = np.stack((a_np, b_np), axis=0)
    assert c_idrak_axis0.shape == c_np_axis0.shape
    assert np.allclose(c_idrak_axis0.data, c_np_axis0)

    # Stack along axis 1
    c_idrak_axis1 = stack([a_idrak, b_idrak], axis=1)
    c_idrak_axis1.realize()
    c_np_axis1 = np.stack((a_np, b_np), axis=1)
    assert c_idrak_axis1.shape == c_np_axis1.shape
    assert np.allclose(c_idrak_axis1.data, c_np_axis1)

"""
def test_softmax():
    a_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2, 3))

    # Softmax along dim -1 (last dimension)
    b_idrak = softmax(a_idrak, dim=-1)
    b_idrak.realize()
    b_np = np.exp(a_np) / np.sum(np.exp(a_np), axis=-1, keepdims=True)

    assert b_idrak.shape == b_np.shape
    assert np.allclose(b_idrak.data, b_np, rtol=1e-4, atol=1e-6)

    # Softmax along dim 0
    b_idrak_dim0 = softmax(a_idrak, dim=0)
    b_idrak_dim0.realize()
    b_np_dim0 = np.exp(a_np) / np.sum(np.exp(a_np), axis=0, keepdims=True)

    assert b_idrak_dim0.shape == b_np_dim0.shape
    assert np.allclose(b_idrak_dim0.data, b_np_dim0, rtol=1e-4, atol=1e-6)

def test_log_softmax():
    a_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2, 3))

    # Log Softmax along dim -1 (last dimension)
    b_idrak = log_softmax(a_idrak, dim=-1)
    b_idrak.realize()
    log_softmax_np = a_np - np.max(a_np, axis=-1, keepdims=True) - np.log(np.sum(np.exp(a_np - np.max(a_np, axis=-1, keepdims=True)), axis=-1, keepdims=True))

    assert b_idrak.shape == log_softmax_np.shape
    assert np.allclose(b_idrak.data, log_softmax_np, rtol=1e-4, atol=1e-6)

    # Log Softmax along dim 0
    b_idrak_dim0 = log_softmax(a_idrak, dim=0)
    b_idrak_dim0.realize()
    log_softmax_np_dim0 = a_np - np.max(a_np, axis=0, keepdims=True) - np.log(np.sum(np.exp(a_np - np.max(a_np, axis=0, keepdims=True)), axis=0, keepdims=True))

    assert b_idrak_dim0.shape == log_softmax_np_dim0.shape
    assert np.allclose(b_idrak_dim0.data, log_softmax_np_dim0, rtol=1e-4, atol=1e-6)

def test_conv2d():
    # Input: (batch_size, in_channels, height, width)
    input_np = np.array([[[[1., 2., 3.],
                            [4., 5., 6.],
                            [7., 8., 9.]]]], dtype=np.float32)
    input_idrak = Tensor(data=input_np, shape=(1, 1, 3, 3))

    # Kernel: (out_channels, in_channels, kernel_height, kernel_width)
    kernel_np = np.array([[[[1., 0.],
                             [0., 1.]]]], dtype=np.float32)
    kernel_idrak = Tensor(data=kernel_np, shape=(1, 1, 2, 2))

    kernel_size = (2, 2)
    stride = (1, 1)
    padding = 0

    output_idrak = conv2d(input_idrak, kernel_idrak, kernel_size, stride, padding)
    output_idrak.realize()

    # Manual convolution for comparison
    # For a simple 1-channel input and 1-channel output, this is equivalent to scipy.signal.convolve2d
    # but we'll do it manually to avoid external dependencies.
    # Output shape calculation: (H_in - K_h + 2*P)/S + 1
    # (3 - 2 + 2*0)/1 + 1 = 2
    # Output shape: (1, 1, 2, 2)
    expected_output_np = np.array([[[[6., 8.],
                                      [12., 14.]]]], dtype=np.float32)

    assert output_idrak.shape == expected_output_np.shape
    assert np.allclose(output_idrak.data, expected_output_np, rtol=1e-4, atol=1e-6)
"""
