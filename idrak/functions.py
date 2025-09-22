from idrak.core.tensor import Tensor
from idrak.ops.uop import *
from idrak.ops.bop import *
from idrak.ops.mop import *
from idrak.ops.rop import *
from idrak.ops.iop import *
from idrak.ops.bop import Conv2D
from idrak.idrak_bindings.c_wrapper_functions import c_zeros, c_ones, c_randn, c_uniform

# =========== Initialization Operations ============
def zeros(shape: tuple[int, ...] | list[int], device: str = "cpu", requires_grad: bool = True) -> Tensor: return Zeros.create_node(shape, requires_grad)

def ones(shape: tuple[int, ...] | list[int], device: str = "cpu", requires_grad: bool = True,) -> Tensor: return Ones.create_node(shape, requires_grad)

def randn(shape: tuple[int, ...] | list[int], seed: int = 42, device: str = "cpu", requires_grad: bool = True) -> Tensor: return Randn.create_node(shape, requires_grad)

def uniform(shape: tuple[int, ...] | list[int], low: float = 0.0, high: float = 1.0, device: str = "cpu", requires_grad: bool = True) -> Tensor: return Uniform.create_node(shape, requires_grad, low=low, high=high)

# ========== Movement Operations ============
def view(a: Tensor, shape: tuple[int, ...]) -> Tensor: return View.create_node(a, shape=shape)
def unsqueeze(a: Tensor, dim: int = 0) -> Tensor: return Unsqueeze.create_node(a, dim=dim)
def squeeze(a: Tensor, dim: int = 0) -> Tensor: return Squeeze.create_node(a, dim=dim)
def expand(a: Tensor, shape: tuple[int, ...]) -> Tensor: return Expand.create_node(a, shape=shape)
def broadcast(a: Tensor, shape: tuple[int, ...]) -> Tensor: return Broadcast.create_node(a, shape)
def transpose(a: Tensor, n: int, m: int) -> Tensor: return Transpose.create_node(a, n=n, m=m)
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
    return RSub.create_node(b, a)

def div(a: Tensor | float, b: Tensor | float) -> Tensor:
    if isinstance(a, Tensor):
        return Div.create_node(a, b)
    return RDiv.create_node(b, a)



# ========= Reduction Operations ==========
def sum(a: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor: return Sum.create_node(a, dim=dim, keepdim=keepdim)
def mean(a: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor: return Mean.create_node(a, dim=dim, keepdim=keepdim)
def max(a: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor: return Max.create_node(a, dim=dim, keepdim=keepdim)

if __name__ == "__main__":
    from idrak.metrics import bce
    a = ones((2,2))
    b = ones((2,2))

    # FIX: adding two tensors from python returns a segment fault most propable that it's because of broadcasting
    c = add(a, b)

    c.realize()

    print(a);print(c)
