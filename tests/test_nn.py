import pytest
import numpy as np
from idrak.core.tensor import Tensor
from idrak.nn.init import xavier_uniform_, xavier_normal_, kaiming_uniform_, kaiming_normal_
from idrak.nn.linear import Linear
from idrak.nn.conv import Conv2d
from idrak.nn.activations import ReLU, Tanh, Sigmoid, Softmax
from idrak.nn.sequential import Sequential

def test_xavier_uniform_():
    shape = (10, 5)
    in_features = 5
    out_features = 10
    weights = xavier_uniform_(shape, in_features, out_features)
    assert weights.shape == shape
    assert weights.requires_grad is True
    # Check bounds (approximate for uniform distribution)
    bound = np.sqrt(6 / (in_features + out_features))
    weights.realize()
    assert np.all(weights.data >= -bound)
    assert np.all(weights.data <= bound)

def test_xavier_normal_():
    shape = (10, 5)
    in_features = 5
    out_features = 10
    weights = xavier_normal_(shape, in_features, out_features)
    assert weights.shape == shape
    assert weights.requires_grad is True
    # Check properties of normal distribution (approximate)
    weights.realize()
    assert np.isclose(np.mean(weights.data), 0.0, atol=0.1) # Mean should be close to 0
    expected_std = np.sqrt(2 / (in_features + out_features))
    assert np.isclose(np.std(weights.data), expected_std, atol=0.1) # Std should be close to expected

def test_kaiming_uniform_():
    shape = (10, 5)
    in_features = 5
    weights = kaiming_uniform_(shape, in_features)
    assert weights.shape == shape
    assert weights.requires_grad is True
    # Check bounds (approximate for uniform distribution)
    bound = np.sqrt(6 / in_features)
    weights.realize()
    assert np.all(weights.data >= -bound)
    assert np.all(weights.data <= bound)

def test_kaiming_normal_():
    shape = (10, 5)
    in_features = 5
    weights = kaiming_normal_(shape, in_features)
    assert weights.shape == shape
    assert weights.requires_grad is True
    # Check properties of normal distribution (approximate)
    weights.realize()
    assert np.isclose(np.mean(weights.data), 0.0, atol=0.1) # Mean should be close to 0
    expected_std = np.sqrt(2 / in_features)
    assert np.isclose(np.std(weights.data), expected_std, atol=0.1) # Std should be close to expected

def test_linear_init():
    in_features = 10
    out_features = 5

    # Test with bias
    linear_layer = Linear(in_features, out_features, bias=True)
    assert isinstance(linear_layer.W, Tensor)
    assert linear_layer.W.shape == (out_features, in_features)
    assert linear_layer.W.requires_grad is True
    assert isinstance(linear_layer.B, Tensor)
    assert linear_layer.B.shape == (1, out_features)
    assert linear_layer.B.requires_grad is True

    # Test without bias
    linear_layer_no_bias = Linear(in_features, out_features, bias=False)
    assert isinstance(linear_layer_no_bias.W, Tensor)
    assert linear_layer_no_bias.W.shape == (out_features, in_features)
    assert linear_layer_no_bias.W.requires_grad is True
    assert linear_layer_no_bias.B is None

def test_linear_forward():
    in_features = 3
    out_features = 2
    batch_size = 4

    linear_layer = Linear(in_features, out_features, bias=True)

    # Manually set weights and bias for predictable results
    linear_layer.W = Tensor(data=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32), shape=(out_features, in_features), requires_grad=True)
    linear_layer.B = Tensor(data=np.array([[0.0, 1.0]], dtype=np.float32), shape=(1, out_features), requires_grad=True)

    input_np = np.array([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0],
                         [7.0, 8.0, 9.0],
                         [10.0, 11.0, 12.0]], dtype=np.float32)
    input_idrak = Tensor(data=input_np, shape=(batch_size, in_features))

    output_idrak = linear_layer(input_idrak)
    output_idrak.realize()

    # Manual forward pass using numpy
    expected_output_np = input_np @ linear_layer.W.data.T + linear_layer.B.data

    assert output_idrak.shape == expected_output_np.shape
    assert np.allclose(output_idrak.data, expected_output_np, rtol=1e-4, atol=1e-6)

    # Test without bias
    linear_layer_no_bias = Linear(in_features, out_features, bias=False)
    linear_layer_no_bias.W = Tensor(data=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32), shape=(out_features, in_features), requires_grad=True)

    output_idrak_no_bias = linear_layer_no_bias(input_idrak)
    output_idrak_no_bias.realize()

    expected_output_np_no_bias = input_np @ linear_layer_no_bias.W.data.T

    assert output_idrak_no_bias.shape == expected_output_np_no_bias.shape
    assert np.allclose(output_idrak_no_bias.data, expected_output_np_no_bias, rtol=1e-4, atol=1e-6)

def test_conv2d_init():
    in_channels = 3
    out_channels = 5
    kernel_size = (3, 3)

    # Test with bias
    conv_layer = Conv2d(in_channels, out_channels, kernel_size, bias=True)
    assert isinstance(conv_layer.weights, Tensor)
    assert conv_layer.weights.shape == (out_channels, in_channels, *kernel_size)
    assert conv_layer.weights.requires_grad is True
    assert isinstance(conv_layer.bias, Tensor)
    assert conv_layer.bias.shape == (out_channels,)
    assert conv_layer.bias.requires_grad is True

    # Test without bias
    conv_layer_no_bias = Conv2d(in_channels, out_channels, kernel_size, bias=False)
    assert isinstance(conv_layer_no_bias.weights, Tensor)
    assert conv_layer_no_bias.weights.shape == (out_channels, in_channels, *kernel_size)
    assert conv_layer_no_bias.weights.requires_grad is True
    assert conv_layer_no_bias.bias is None

def test_conv2d_forward_shape():
    in_channels = 1
    out_channels = 1
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = 0

    conv_layer = Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

    input_np = np.zeros((1, in_channels, 5, 5), dtype=np.float32)
    input_idrak = Tensor(data=input_np, shape=(1, in_channels, 5, 5))

    output_idrak = conv_layer(input_idrak)
    # Calculate expected output shape: (H_in - K_h + 2*P)/S + 1
    # H_out = (5 - 3 + 2*0)/1 + 1 = 3
    expected_h = (5 - kernel_size[0] + 2 * padding) // stride[0] + 1
    expected_w = (5 - kernel_size[1] + 2 * padding) // stride[1] + 1
    expected_shape = (1, out_channels, expected_h, expected_w)

    assert output_idrak.shape == expected_shape

    # Test with different stride and padding
    stride = (2, 2)
    padding = 1
    conv_layer_2 = Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    output_idrak_2 = conv_layer_2(input_idrak)
    expected_h_2 = (5 - kernel_size[0] + 2 * padding) // stride[0] + 1
    expected_w_2 = (5 - kernel_size[1] + 2 * padding) // stride[1] + 1
    expected_shape_2 = (1, out_channels, expected_h_2, expected_w_2)
    assert output_idrak_2.shape == expected_shape_2

def test_relu_activation():
    relu_module = ReLU()
    input_np = np.array([[-1.0, 0.0, 1.0], [-2.0, 3.0, -4.0]], dtype=np.float32)
    input_idrak = Tensor(data=input_np, shape=(2, 3))

    output_idrak = relu_module(input_idrak)
    output_idrak.realize()

    expected_output_np = np.maximum(0, input_np)

    assert output_idrak.shape == expected_output_np.shape
    assert np.allclose(output_idrak.data, expected_output_np)

def test_tanh_activation():
    tanh_module = Tanh()
    input_np = np.array([[-1.0, 0.0, 1.0], [-2.0, 3.0, -4.0]], dtype=np.float32)
    input_idrak = Tensor(data=input_np, shape=(2, 3))

    output_idrak = tanh_module(input_idrak)
    output_idrak.realize()

    expected_output_np = np.tanh(input_np)

    assert output_idrak.shape == expected_output_np.shape
    assert np.allclose(output_idrak.data, expected_output_np, rtol=1e-4, atol=1e-6)

def test_sigmoid_activation():
    sigmoid_module = Sigmoid()
    input_np = np.array([[-1.0, 0.0, 1.0], [-2.0, 3.0, -4.0]], dtype=np.float32)
    input_idrak = Tensor(data=input_np, shape=(2, 3))

    output_idrak = sigmoid_module(input_idrak)
    output_idrak.realize()

    expected_output_np = 1 / (1 + np.exp(-input_np))

    assert output_idrak.shape == expected_output_np.shape
    assert np.allclose(output_idrak.data, expected_output_np, rtol=1e-4, atol=1e-6)

def test_softmax_activation():
    softmax_module = Softmax()
    input_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_idrak = Tensor(data=input_np, shape=(2, 3))

    # Softmax along dim -1 (default)
    output_idrak = softmax_module(input_idrak)
    output_idrak.realize()

    expected_output_np = np.exp(input_np) / np.sum(np.exp(input_np), axis=-1, keepdims=True)

    assert output_idrak.shape == expected_output_np.shape
    assert np.allclose(output_idrak.data, expected_output_np, rtol=1e-4, atol=1e-6)

def test_sequential_init():
    linear1 = Linear(10, 5)
    relu = ReLU()
    linear2 = Linear(5, 2)
    seq_model = Sequential(linear1, relu, linear2)

    assert len(seq_model.layers) == 3
    assert seq_model.layers[0] == linear1
    assert seq_model.layers[1] == relu
    assert seq_model.layers[2] == linear2

def test_sequential_forward():
    in_features = 10
    hidden_features = 5
    out_features = 2
    batch_size = 4

    linear1 = Linear(in_features, hidden_features, bias=True)
    relu = ReLU()
    linear2 = Linear(hidden_features, out_features, bias=True)
    seq_model = Sequential(linear1, relu, linear2)

    # Manually set weights and biases for predictable results
    linear1.W = Tensor(data=np.full((hidden_features, in_features), 0.1, dtype=np.float32), shape=(hidden_features, in_features), requires_grad=True)
    linear1.B = Tensor(data=np.full((1, hidden_features), 0.0, dtype=np.float32), shape=(1, hidden_features), requires_grad=True)
    linear2.W = Tensor(data=np.full((out_features, hidden_features), 0.2, dtype=np.float32), shape=(out_features, hidden_features), requires_grad=True)
    linear2.B = Tensor(data=np.full((1, out_features), 0.0, dtype=np.float32), shape=(1, out_features), requires_grad=True)

    input_np = np.full((batch_size, in_features), 1.0, dtype=np.float32)
    input_idrak = Tensor(data=input_np, shape=(batch_size, in_features))

    output_idrak = seq_model(input_idrak)
    # output_idrak.realize() # Sequential.forward already calls realize()

    # Manual forward pass using numpy
    # Layer 1: Linear
    out_linear1_np = input_np @ linear1.W.data.T + linear1.B.data
    # Layer 2: ReLU
    out_relu_np = np.maximum(0, out_linear1_np)
    # Layer 3: Linear
    expected_output_np = out_relu_np @ linear2.W.data.T + linear2.B.data

    assert output_idrak.shape == expected_output_np.shape
    assert np.allclose(output_idrak.data, expected_output_np, rtol=1e-4, atol=1e-6)

def test_sequential_params_buffers():
    linear1 = Linear(10, 5)
    relu = ReLU()
    linear2 = Linear(5, 2, bias=False)
    seq_model = Sequential(linear1, relu, linear2)

    params = seq_model.params
    buffers = seq_model.buffers

    # Linear layers have weights (W) and optionally bias (B) as parameters
    # ReLU has no parameters
    # Total parameters: linear1.W, linear1.B, linear2.W
    assert len(params) == 3
    assert linear1.W in params
    assert linear1.B in params
    assert linear2.W in params

    # Buffers are tensors that do not require gradients. In this case, there are none by default.
    assert len(buffers) == 0
