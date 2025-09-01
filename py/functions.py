from py.core.tensor import Tensor
from py.ops.uop import *
from py.ops.bop import *
from py.ops.mop import *
from py.ops.rop import *
from py.elnawah_bindings.c_wrapper_functions import c_zeros, c_ones, c_randn, c_uniform

# =========== Unary Operations =============
def relu(a: Tensor) -> Tensor: return ReLU.create_node(a)
def log(a: Tensor) -> Tensor: return Log.create_node(a)
def exp(a: Tensor) -> Tensor: return Exp.create_node(a)
def abs(a: Tensor) -> Tensor: return Abs.create_node(a)
def neg(a: Tensor) -> Tensor: return Neg.create_node(a)

# =========== Binary Operations =============
def add(a: Tensor | float, b: Tensor | float) -> Tensor: return Add.create_node(a, b)
def mul(a: Tensor | float, b: Tensor | float) -> Tensor: return Mul.create_node(a, b)

def sub(a: Tensor | float, b: Tensor | float) -> Tensor:
    if isinstance(a, Tensor):
        return Sub.create_node(a, b)
    return RSub.create_node(b, a)

def div(a: Tensor | float, b: Tensor | float) -> Tensor:
    if isinstance(a, Tensor):
        return Div.create_node(a, b)
    return RDiv.create_node(b, a)


# ========== Movment Operations ============
def view(a: Tensor, shape: tuple[int, ...]) -> Tensor: return View.create_node(a, shape=shape)
def unsqueeze(a: Tensor, dim: int = 0) -> Tensor: return Unsqueeze.create_node(a, dim=dim)
def squeeze(a: Tensor, dim: int = 0) -> Tensor: return Squeeze.create_node(a, dim=dim)
def expand(a: Tensor, shape: tuple[int, ...]) -> Tensor: return Expand.create_node(a, shape=shape)
def broadcast(a: Tensor, shape: tuple[int, ...]) -> Tensor: return Broadcast.create_node(a, shape=shape, ndim=len(shape))
def transpose(a: Tensor, n: int, m: int) -> Tensor: return Transpose.create_node(a, n=n, m=m)

# ========= Reduction Operations ==========
def sum(a: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor: return Sum.create_node(a, dim=dim, keepdim=keepdim)
def mean(a: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor: return Mean.create_node(a, dim=dim, keepdim=keepdim)
def max(a: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor: return Max.create_node(a, dim=dim, keepdim=keepdim)
