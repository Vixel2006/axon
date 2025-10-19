import numpy as np

from axon.axon_bindings.ctypes_definitions import CStorage
from axon.core.tensor import Tensor
from axon.core.device import Device
from axon.ops.uop import *
from axon.ops.bop import *
from axon.ops.mop import *
from axon.ops.rop import *

from axon.ops.bop import Conv2D
from axon.axon_bindings.c_wrapper_functions import c_zeros, c_ones, c_randn, c_uniform, c_from_data
import ctypes

# =========== Initialization Operations ============
def zeros(shape: tuple[int, ...] | list[int], device: str | Device = "cpu", requires_grad: bool = True) -> Tensor:
    if isinstance(device, str):
        device = Device.parse(device)
    out = Tensor(shape=shape, device=device, requires_grad=requires_grad)
    c_zeros(out.c_tensor_ptr)
    return out

def ones(shape: tuple[int, ...] | list[int], device: str | Device = "cpu", requires_grad: bool = True) -> Tensor:
    if isinstance(device, str):
        device = Device.parse(device)
    out = Tensor(shape=shape, device=device, requires_grad=requires_grad)
    c_ones(out.c_tensor_ptr)
    return out

def randn(shape: tuple[int, ...] | list[int], seed: int = 42, device: str | Device = "cpu", requires_grad: bool = True) -> Tensor:
    if isinstance(device, str):
        device = Device.parse(device)
    out = Tensor(shape=shape, device=device, requires_grad=requires_grad)
    # NOTE: We need to add the seed to the randn function
    c_randn(out.c_tensor_ptr)
    return out

def uniform(shape: tuple[int, ...] | list[int], low: float = 0.0, high: float = 1.0, device: str | Device = "cpu", requires_grad: bool = True) -> Tensor:
    if isinstance(device, str):
        device = Device.parse(device)
    out = Tensor(shape=shape, device=device, requires_grad=requires_grad)
    c_uniform(out.c_tensor_ptr, low, high)
    return out

def from_data(shape: tuple[int, ...] | list[int], data: list[int] | list[float] | np.ndarray, device: str | Device = "cpu", requires_grad: bool = True) -> Tensor:
    if isinstance(device, str):
        device = Device.parse(device)
    out = Tensor(shape=shape, device=device, requires_grad=requires_grad)

    if isinstance(data, np.ndarray):
        out._numpy_data_ref = data.astype(np.float32)
    elif isinstance(data, (list, tuple)):
        out._numpy_data_ref = np.array(data, dtype=np.float32)
    else:
        raise TypeError(f"Unsupported data type for from_data: {type(data)}. Expected list, tuple, or numpy.ndarray.")

    data_ptr = out._numpy_data_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_from_data(out.c_tensor_ptr, data_ptr)
    return out


def from_c_storage(shape: tuple[int], c_storage_ptr: ctypes.POINTER(CStorage), device: str | Device = "cpu", requires_grad: bool = False) -> Tensor:
    if isinstance(device, str):
        device = Device.parse(device)
    out = Tensor(shape=shape, device=device, requires_grad=requires_grad)
    # Assign the provided CStorage pointer directly to the Tensor's data
    out.c_tensor_ptr.contents.data = c_storage_ptr
    # Increment the reference counter for the CStorage
    if c_storage_ptr:
        c_storage_ptr.contents.counter += 1
    return out


# ========== Movement Operations ============
def view(a: Tensor, shape: tuple[int, ...]) -> Tensor: return View.create_node(a, shape)
def unsqueeze(a: Tensor, dim: int = 0) -> Tensor: return Unsqueeze.create_node(a, dim)
def squeeze(a: Tensor, dim: int = 0) -> Tensor: return Squeeze.create_node(a, dim)
def expand(a: Tensor, shape: tuple[int, ...]) -> Tensor: return Expand.create_node(a, shape)
def broadcast(a: Tensor, shape: tuple[int, ...]) -> Tensor: return Broadcast.create_node(a, shape)
def transpose(a: Tensor, n: int, m: int) -> Tensor: return Transpose.create_node(a, n, m)
def concat(a: list[Tensor], axis: int = 0) -> Tensor: return Concat.create_node(a, axis=axis)
def stack(a: list[Tensor], axis: int = 0) -> Tensor: return Stack.create_node(a, axis=axis)

# =========== Unary Operations =============
def relu(a: Tensor) -> Tensor: return ReLU.create_node(a)
def clip(a: Tensor, min_val: float, max_val: float) -> Tensor: return Clip.create_node(a, min_val=min_val, max_val=max_val)
def log(a: Tensor) -> Tensor: return Log.create_node(a)
def exp(a: Tensor) -> Tensor: return Exp.create_node(a)
def abs(a: Tensor) -> Tensor: return Abs.create_node(a)
def neg(a: Tensor) -> Tensor: return Neg.create_node(a)

# =========== Binary Operations =============
def add(a: Tensor | float, b: Tensor | float) -> Tensor: return Add.create_node(a, b)
def mul(a: Tensor | float, b: Tensor | float) -> Tensor: return Mul.create_node(a, b)
def pow(a: Tensor, b: Tensor | float) -> Tensor: return Pow.create_node(a, b)
def matmul(a: Tensor, b: Tensor) -> Tensor: return MatMul.create_node(a, b)
def dot(a: Tensor, b: Tensor) -> Tensor: return Dot.create_node(a, b)

def conv2d(a: Tensor, b: Tensor, kernel_size: tuple[int, ...], stride: tuple[int, int], padding: int) -> Tensor: return Conv2D.create_node(a, b, kernel_size=kernel_size, stride=stride, padding=padding)


def sub(a: Tensor | float, b: Tensor | float) -> Tensor:
    if isinstance(a, Tensor):
        return Sub.create_node(a, b)
    return RSub.create_node(a, b)

def div(a: Tensor | float, b: Tensor | float) -> Tensor:
    if isinstance(a, Tensor):
        return Div.create_node(a, b)
    return RDiv.create_node(a, b)

def softmax(a: Tensor, dim: int = -1) -> Tensor:
    x_max = max(a, dim=dim, keepdim=True)
    exp_x = (a - x_max).exp()
    return exp_x / sum(exp_x, dim=dim, keepdim=True)

def log_softmax(a: Tensor, dim: int = -1) -> Tensor:
    x_max = max(a, dim=dim, keepdim=True)
    shifted = a - x_max
    log_sum_exp = log(sum(exp(shifted), dim=dim, keepdim=True))
    return shifted - log_sum_exp

# ========= Reduction Operations ==========
def sum(a: Tensor, dim: int | None = None, keepdim: bool = True) -> Tensor: return Sum.create_node(a, dim=dim, keepdim=keepdim)
def mean(a: Tensor, dim: int | None = None, keepdim: bool = True) -> Tensor: return Mean.create_node(a, dim=dim, keepdim=keepdim)
def max(a: Tensor, dim: int | None = None, keepdim: bool = True) -> Tensor: return Max.create_node(a, dim=dim, keepdim=keepdim)

if __name__ == "__main__":
    a = from_data((2, 2, 2), [[[1, 2], [2, 3]], [[3, 4], [4, 5]]], device="cpu")
    
    a_grad_flattened = a.grad.flatten()

    a_grad_flattened.realize()
    print(a_grad_flattened)
    #b = from_data((2, 2, 2), [[[1, 2], [2, 3]], [[3, 4], [4, 5]]], device="cpu")
