from py.core.tensor import Tensor
from py.ops.functions.binary_ops import Add, Sub, RSub, Mul, Div, RDiv, MatMul
from py.ops.functions.unary_ops import ReLU, Log, Exp, Softmax, Abs, Neg
from py.ops.functions.reduction_ops import Sum, Mean, Max
from py.ops.functions.movement_ops import View, Unsqueeze, Squeeze, Transpose, Expand, Broadcast
from py.elnawah_bindings.c_wrapper_functions import c_zeros, c_ones, c_randn, c_uniform

def add(a: Tensor, b: Tensor) -> Tensor:
    return Add.apply(a, b)

def sub(a: Tensor, b: Tensor) -> Tensor:
    return Sub.apply(a, b)

def rsub(a: Tensor, b: float) -> Tensor:
    return RSub.apply(a, b)

def mul(a: Tensor, b: Tensor) -> Tensor:
    return Mul.apply(a, b)

def div(a: Tensor, b: Tensor) -> Tensor:
    return Div.apply(a, b)

def rdiv(a: Tensor, b: float) -> Tensor:
    return RDiv.apply(a, b)

def matmul(a: Tensor, b: Tensor) -> Tensor:
    return MatMul.apply(a, b)

def relu(a: Tensor) -> Tensor:
    return ReLU.apply(a)

def log(a: Tensor) -> Tensor:
    return Log.apply(a)

def exp(a: Tensor) -> Tensor:
    return Exp.apply(a)

def softmax(a: Tensor) -> Tensor:
    return Softmax.apply(a)

def abs(a: Tensor) -> Tensor:
    return Abs.apply(a)

def neg(a: Tensor) -> Tensor:
    return Neg.apply(a)

def sum(a: Tensor, axis: int = 0, keepdim: bool = True) -> Tensor:
    return Sum.apply(a, axis=axis, keepdim=keepdim)

def mean(a: Tensor, axis: int = 0, keepdim: bool = True) -> Tensor:
    return Mean.apply(a, axis=axis, keepdim=keepdim)

def max(a: Tensor, axis: int = 0, keepdim: bool = True) -> Tensor:
    return Max.apply(a, axis=axis, keepdim=keepdim)

def view(a: Tensor, shape: list) -> Tensor:
    return View.apply(a, shape=shape)

def unsqueeze(a: Tensor, dim: int = 0) -> Tensor:
    return Unsqueeze.apply(a, dim=dim)

def squeeze(a: Tensor, dim: int = 0) -> Tensor:
    return Squeeze.apply(a, dim=dim)

def transpose(a: Tensor, n: int = -2, m: int = -1) -> Tensor:
    return Transpose.apply(a, n=n, m=m)

def expand(a: Tensor, shape: list[int]) -> Tensor:
    return Expand.apply(a, shape=shape)

def broadcast(a: Tensor, shape: list[int]) -> Tensor:
    ndim = len(shape)
    return Broadcast.apply(a, shape=shape, ndim=ndim)

def zeros(shape: tuple, requires_grad: bool = True) -> Tensor:
    ndim = len(shape)
    c_tensor_ptr = c_zeros(shape, ndim, requires_grad)
    return Tensor(_c_tensor_ptr=c_tensor_ptr)

def ones(shape: tuple, requires_grad: bool = True) -> Tensor:
    ndim = len(shape)
    c_tensor_ptr = c_ones(shape, ndim, requires_grad)
    return Tensor(_c_tensor_ptr=c_tensor_ptr)

def randn(shape: tuple, seed: int, requires_grad: bool = True) -> Tensor:
    ndim = len(shape)
    c_tensor_ptr = c_randn(shape, ndim, seed, requires_grad)
    return Tensor(_c_tensor_ptr=c_tensor_ptr)

def uniform(shape: tuple, low: int, high: int, requires_grad: bool = True) -> Tensor:
    ndim = len(shape)
    c_tensor_ptr = c_uniform(shape, ndim, low, high, requires_grad)
    return Tensor(_c_tensor_ptr=c_tensor_ptr)


if __name__ == "__main__":
    n = uniform((2,4), 0, 1)

    print(n)