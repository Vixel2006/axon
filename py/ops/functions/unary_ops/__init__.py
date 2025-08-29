from __future__ import annotations
import ctypes
from typing import Any, Tuple, Optional, List
import sys

from py.elnawah_bindings.c_wrapper_functions import (
    c_relu,
    c_log,
    c_exp,
    c_softmax,
    c_abs,
    c_neg,
    c_relu_grad_op,
    c_log_grad_op,
    c_exp_grad_op,
    c_abs_grad_op,
    c_neg_grad_op,
)
from py.elnawah_bindings.ctypes_definitions import CTensor, BackwardFnType
from py.elnawah_bindings.c_library_loader import tensor_lib
from py.ops.lazy_ops.unary_ops import (
    LazyReLU,
    LazyLog,
    LazyExp,
    LazySoftmax,
    LazyAbs,
    LazyNeg,
)
from py.ops.base import Function


class ReLU(Function):
    """
    Rectified Linear Unit (ReLU).

    Forward:
        out = max(0, a)

    Backward:
        dL/da = dL/dout if a > 0 else 0

    Notes:
        Uses efficient C SIMD kernels for forward and backward.
    """

    lazy_op_class = LazyReLU

    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.relu_grad_op)

    def _execute_forward(self, out_tensor: "Tensor", a: "Tensor") -> "Tensor":
        from py.core.tensor import Tensor

        c_relu(a._c_tensor, out_tensor._c_tensor)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        c_relu_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)


class Log(Function):
    """
    Natural logarithm.

    Forward:
        out = log(a)

    Backward:
        dL/da = (1 / a) * dL/dout

    Notes:
        - Uses C-level SIMD kernel `c_log`.
        - Registers C-level backward function `c_log_grad_op`.
    """

    lazy_op_class = LazyLog

    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.log_grad_op)

    def _execute_forward(self, out_tensor: "Tensor", a: "Tensor") -> "Tensor":
        from py.core.tensor import Tensor

        c_log(a._c_tensor, out_tensor._c_tensor)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        c_log_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)


class Exp(Function):
    """
    Exponential function.

    Forward:
        out = exp(a)

    Backward:
        dL/da = exp(a) * dL/dout

    Notes:
        - Uses C-level SIMD kernel `c_exp`.
        - Registers C-level backward function `c_exp_grad_op`.
    """

    lazy_op_class = LazyExp

    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.exp_grad_op)

    def _execute_forward(self, out_tensor: "Tensor", a: "Tensor") -> "Tensor":
        from py.core.tensor import Tensor

        c_exp(a._c_tensor, out_tensor._c_tensor)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        c_exp_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)


class Softmax(Function):
    """
    Softmax activation (over the last dimension).

    Forward:
        out[i] = exp(a[i]) / Σ_j exp(a[j])

    Backward:
        dL/da = out * (dL/dout - Σ_j(dL/dout_j * out_j))

    Notes:
        - Uses C-level kernel `c_softmax`.
        - Backward not yet implemented (TODO: `c_softmax_grad_op`).
    """

    lazy_op_class = LazySoftmax

    def _execute_forward(self, out_tensor: "Tensor", a: "Tensor") -> "Tensor":
        from py.core.tensor import Tensor

        c_softmax(a._c_tensor, out_tensor._c_tensor)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        # Softmax backward is more complex, will need to implement c_softmax_grad_op
        pass  # TODO: Implement softmax_grad_op


class Abs(Function):
    """
    Absolute value.

    Forward:
        out = |a|

    Backward:
        dL/da = sign(a) * dL/dout
        where sign(a) = +1 if a > 0, -1 if a < 0, else 0

    Notes:
        - Uses C-level SIMD kernel `c_abs`.
        - Registers C-level backward function `c_abs_grad_op`.
    """

    lazy_op_class = LazyAbs

    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.abs_grad_op)

    def _execute_forward(self, out_tensor: "Tensor", a: "Tensor") -> "Tensor":
        from py.core.tensor import Tensor

        c_abs(a._c_tensor, out_tensor._c_tensor)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        c_abs_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)


class Neg(Function):
    """
    Elementwise negation.

    Forward:
        out = -a

    Backward:
        dL/da = -dL/dout

    Notes:
        - Uses C-level SIMD kernel `c_neg`.
        - Registers C-level backward function `c_neg_grad_op`.
    """

    lazy_op_class = LazyNeg

    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.neg_grad_op)

    def _execute_forward(self, out_tensor: "Tensor", a: "Tensor") -> "Tensor":
        from py.core.tensor import Tensor

        c_neg(a._c_tensor, out_tensor._c_tensor)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        c_neg_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)
