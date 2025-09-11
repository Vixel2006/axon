import pytest
import numpy as np
from idrak.core.tensor import Tensor
from idrak.ops.bop import Add, Mul, Sub, Div, Pow, MatMul, Dot, Conv2D
from idrak.ops.uop import Neg, ReLU, Log, Exp, Abs
from idrak.ops.mop import View, Unsqueeze, Squeeze, Expand, Broadcast, Transpose, Concat, Stack
from idrak.ops.rop import Sum, Mean, Max

def test_add_create_node_shape():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    a_idrak = Tensor(data=a_np, shape=(2, 2))
    b_idrak = Tensor(data=b_np, shape=(2, 2))

    # Test with two tensors
    result_tensor = Add.create_node(a_idrak, b_idrak)
    expected_shape = np.add(a_np, b_np).shape
    assert result_tensor.shape == expected_shape

    # Test with tensor and scalar
    scalar = 5.0
    result_tensor_scalar = Add.create_node(a_idrak, scalar)
    expected_shape_scalar = np.add(a_np, scalar).shape
    assert result_tensor_scalar.shape == expected_shape_scalar

def test_mul_create_node_shape():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    a_idrak = Tensor(data=a_np, shape=(2, 2))
    b_idrak = Tensor(data=b_np, shape=(2, 2))

    # Test with two tensors
    result_tensor = Mul.create_node(a_idrak, b_idrak)
    expected_shape = np.multiply(a_np, b_np).shape
    assert result_tensor.shape == expected_shape

    # Test with tensor and scalar
    scalar = 5.0
    result_tensor_scalar = Mul.create_node(a_idrak, scalar)
    expected_shape_scalar = np.multiply(a_np, scalar).shape
    assert result_tensor_scalar.shape == expected_shape_scalar

def test_sub_create_node_shape():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    a_idrak = Tensor(data=a_np, shape=(2, 2))
    b_idrak = Tensor(data=b_np, shape=(2, 2))

    result_tensor = Sub.create_node(a_idrak, b_idrak)
    expected_shape = np.subtract(a_np, b_np).shape
    assert result_tensor.shape == expected_shape

    scalar = 5.0
    result_tensor_scalar = Sub.create_node(a_idrak, scalar)
    expected_shape_scalar = np.subtract(a_np, scalar).shape
    assert result_tensor_scalar.shape == expected_shape_scalar

def test_div_create_node_shape():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    a_idrak = Tensor(data=a_np, shape=(2, 2))
    b_idrak = Tensor(data=b_np, shape=(2, 2))

    result_tensor = Div.create_node(a_idrak, b_idrak)
    expected_shape = np.divide(a_np, b_np).shape
    assert result_tensor.shape == expected_shape

    scalar = 5.0
    result_tensor_scalar = Div.create_node(a_idrak, scalar)
    expected_shape_scalar = np.divide(a_np, scalar).shape
    assert result_tensor_scalar.shape == expected_shape_scalar

def test_pow_create_node_shape():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.array([[2.0, 3.0], [1.0, 2.0]], dtype=np.float32)

    a_idrak = Tensor(data=a_np, shape=(2, 2))
    b_idrak = Tensor(data=b_np, shape=(2, 2))

    result_tensor = Pow.create_node(a_idrak, b_idrak)
    expected_shape = np.power(a_np, b_np).shape
    assert result_tensor.shape == expected_shape

    scalar = 2.0
    result_tensor_scalar = Pow.create_node(a_idrak, scalar)
    expected_shape_scalar = np.power(a_np, scalar).shape
    assert result_tensor_scalar.shape == expected_shape_scalar

def test_matmul_create_node_shape():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    a_idrak = Tensor(data=a_np, shape=(2, 2))
    b_idrak = Tensor(data=b_np, shape=(2, 2))

    result_tensor = MatMul.create_node(a_idrak, b_idrak)
    expected_shape = np.matmul(a_np, b_np).shape
    assert result_tensor.shape == expected_shape

def test_dot_create_node_shape():
    a_np = np.array([1.0, 2.0], dtype=np.float32)
    b_np = np.array([3.0, 4.0], dtype=np.float32)

    a_idrak = Tensor(data=a_np, shape=(2,))
    b_idrak = Tensor(data=b_np, shape=(2,))

    result_tensor = Dot.create_node(a_idrak, b_idrak)
    expected_shape = (1,)
    assert result_tensor.shape == expected_shape

def test_conv2d_create_node_shape():
    input_np = np.zeros((1, 1, 5, 5), dtype=np.float32) # (N, C_in, H_in, W_in)
    kernel_np = np.zeros((1, 1, 3, 3), dtype=np.float32) # (C_out, C_in, K_h, K_w)

    input_idrak = Tensor(data=input_np, shape=(1, 1, 5, 5))
    kernel_idrak = Tensor(data=kernel_np, shape=(1, 1, 3, 3))

    kernel_size = (3, 3)
    stride = (1, 1)
    padding = 0

    result_tensor = Conv2D.create_node(input_idrak, kernel_idrak, kernel_size, stride, padding)

    # Calculate expected output shape manually or using a known formula
    # H_out = (H_in - K_h + 2*P)/S + 1
    # W_out = (W_in - K_w + 2*P)/S + 1
    # For this example: H_out = (5 - 3 + 0)/1 + 1 = 3
    # W_out = (5 - 3 + 0)/1 + 1 = 3
    # Output shape: (N, C_out, H_out, W_out)
    expected_shape = (1, 1, 3, 3)
    assert result_tensor.shape == expected_shape

def test_neg_create_node_shape():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2, 2))

    result_tensor = Neg.create_node(a_idrak)
    expected_shape = (-a_np).shape
    assert result_tensor.shape == expected_shape

def test_relu_create_node_shape():
    a_np = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2, 2))

    result_tensor = ReLU.create_node(a_idrak)
    expected_shape = np.maximum(0, a_np).shape
    assert result_tensor.shape == expected_shape

def test_log_create_node_shape():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2, 2))

    result_tensor = Log.create_node(a_idrak)
    expected_shape = np.log(a_np).shape
    assert result_tensor.shape == expected_shape

def test_exp_create_node_shape():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2, 2))

    result_tensor = Exp.create_node(a_idrak)
    expected_shape = np.exp(a_np).shape
    assert result_tensor.shape == expected_shape

def test_abs_create_node_shape():
    a_np = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2, 2))

    result_tensor = Abs.create_node(a_idrak)
    expected_shape = np.abs(a_np).shape
    assert result_tensor.shape == expected_shape

def test_view_create_node_shape():
    a_np = np.arange(1.0, 7.0).reshape(2, 3).astype(np.float32)
    a_idrak = Tensor(data=a_np, shape=(2, 3))

    new_shape = (3, 2)
    result_tensor = View.create_node(a_idrak, shape=new_shape)
    expected_shape = new_shape
    assert result_tensor.shape == expected_shape

def test_unsqueeze_create_node_shape():
    a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(3,))

    result_tensor = Unsqueeze.create_node(a_idrak, dim=0)
    expected_shape = np.expand_dims(a_np, axis=0).shape
    assert result_tensor.shape == expected_shape

    result_tensor = Unsqueeze.create_node(a_idrak, dim=1)
    expected_shape = np.expand_dims(a_np, axis=1).shape
    assert result_tensor.shape == expected_shape

def test_squeeze_create_node_shape():
    a_np = np.array([[[1.0, 2.0]]], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(1, 1, 2))

    result_tensor = Squeeze.create_node(a_idrak, dim=0)
    expected_shape = np.squeeze(a_np, axis=0).shape
    assert result_tensor.shape == expected_shape

    result_tensor = Squeeze.create_node(a_idrak, dim=1)
    expected_shape = np.squeeze(a_np, axis=1).shape
    assert result_tensor.shape == expected_shape

def test_expand_create_node_shape():
    a_np = np.array([1.0, 2.0], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2,))

    new_shape = (2, 2)
    result_tensor = Expand.create_node(a_idrak, shape=new_shape)
    expected_shape = np.broadcast_to(a_np, new_shape).shape
    assert result_tensor.shape == expected_shape

def test_broadcast_create_node_shape():
    a_np = np.array([1.0, 2.0], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2,))

    new_shape = (2, 2)
    result_tensor = Broadcast.create_node(a_idrak, shape=new_shape, ndim=len(new_shape))
    expected_shape = np.broadcast_to(a_np, new_shape).shape
    assert result_tensor.shape == expected_shape

def test_transpose_create_node_shape():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2, 2))

    result_tensor = Transpose.create_node(a_idrak, n=0, m=1)
    expected_shape = np.transpose(a_np, axes=(1, 0)).shape
    assert result_tensor.shape == expected_shape

    a_np_3d = np.arange(1.0, 28.0).reshape(3, 3, 3).astype(np.float32)
    a_idrak_3d = Tensor(data=a_np_3d, shape=(3, 3, 3))

    result_tensor_3d = Transpose.create_node(a_idrak_3d, n=0, m=2)
    expected_shape_3d = np.transpose(a_np_3d, axes=(2, 1, 0)).shape
    assert result_tensor_3d.shape == expected_shape_3d

def test_concat_create_node_shape():
    a_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b_np = np.array([[5, 6], [7, 8]], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2, 2))
    b_idrak = Tensor(data=b_np, shape=(2, 2))

    result_tensor_axis0 = Concat.create_node([a_idrak, b_idrak], axis=0)
    expected_shape_axis0 = np.concatenate((a_np, b_np), axis=0).shape
    assert result_tensor_axis0.shape == expected_shape_axis0

    result_tensor_axis1 = Concat.create_node([a_idrak, b_idrak], axis=1)
    expected_shape_axis1 = np.concatenate((a_np, b_np), axis=1).shape
    assert result_tensor_axis1.shape == expected_shape_axis1

def test_stack_create_node_shape():
    a_np = np.array([1, 2], dtype=np.float32)
    b_np = np.array([3, 4], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2,))
    b_idrak = Tensor(data=b_np, shape=(2,))

    result_tensor_axis0 = Stack.create_node([a_idrak, b_idrak], axis=0)
    expected_shape_axis0 = np.stack((a_np, b_np), axis=0).shape
    assert result_tensor_axis0.shape == expected_shape_axis0

    result_tensor_axis1 = Stack.create_node([a_idrak, b_idrak], axis=1)
    expected_shape_axis1 = np.stack((a_np, b_np), axis=1).shape
    assert result_tensor_axis1.shape == expected_shape_axis1

def test_sum_create_node_shape():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2, 2))

    result_tensor = Sum.create_node(a_idrak, dim=None, keepdim=False)
    expected_shape = (1,)
    assert result_tensor.shape == expected_shape

    result_tensor_dim0 = Sum.create_node(a_idrak, dim=0, keepdim=False)
    expected_shape_dim0 = np.sum(a_np, axis=0).shape
    assert result_tensor_dim0.shape == expected_shape_dim0

    result_tensor_dim1_keepdim = Sum.create_node(a_idrak, dim=1, keepdim=True)
    expected_shape_dim1_keepdim = np.sum(a_np, axis=1, keepdims=True).shape
    assert result_tensor_dim1_keepdim.shape == expected_shape_dim1_keepdim

def test_mean_create_node_shape():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2, 2))

    result_tensor = Mean.create_node(a_idrak, dim=None, keepdim=False)
    expected_shape = (1,)
    assert result_tensor.shape == expected_shape

    result_tensor_dim0 = Mean.create_node(a_idrak, dim=0, keepdim=False)
    expected_shape_dim0 = np.mean(a_np, axis=0).shape
    assert result_tensor_dim0.shape == expected_shape_dim0

    result_tensor_dim1_keepdim = Mean.create_node(a_idrak, dim=1, keepdim=True)
    expected_shape_dim1_keepdim = np.mean(a_np, axis=1, keepdims=True).shape
    assert result_tensor_dim1_keepdim.shape == expected_shape_dim1_keepdim

def test_max_create_node_shape():
    a_np = np.array([[1.0, 5.0], [3.0, 2.0]], dtype=np.float32)
    a_idrak = Tensor(data=a_np, shape=(2, 2))

    result_tensor = Max.create_node(a_idrak, dim=None, keepdim=False)
    expected_shape = (1,)
    assert result_tensor.shape == expected_shape

    result_tensor_dim0 = Max.create_node(a_idrak, dim=0, keepdim=False)
    expected_shape_dim0 = np.max(a_np, axis=0).shape
    assert result_tensor_dim0.shape == expected_shape_dim0

    result_tensor_dim1_keepdim = Max.create_node(a_idrak, dim=1, keepdim=True)
    expected_shape_dim1_keepdim = np.max(a_np, axis=1, keepdims=True).shape
    assert result_tensor_dim1_keepdim.shape == expected_shape_dim1_keepdim