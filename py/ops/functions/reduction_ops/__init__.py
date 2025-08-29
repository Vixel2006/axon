from __future__ import annotations
import ctypes
from typing import Any, Tuple, Optional, List
import sys

from py.elnawah_bindings.c_wrapper_functions import (
    c_sum,
    c_mean,
    c_max,
    c_sum_grad_op,
    c_mean_grad_op,
    c_max_grad_op,
)
from py.elnawah_bindings.ctypes_definitions import CTensor, BackwardFnType
from py.elnawah_bindings.c_library_loader import tensor_lib
from py.ops.lazy_ops.reduction_ops import LazySum, LazyMean, LazyMax
from py.ops.base import Function


class Sum(Function):
    """
    Reduction: sum over a given axis.

    Forward:
        out = Σ a along `axis`

    Backward:
        dL/da = broadcast(dL/dout, shape of a)

    Notes:
        - Uses C-level kernel `c_sum`.
    """

    lazy_op_class = LazySum

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", axis: int = 0, keepdim: bool = True
    ) -> "Tensor":
        from py.core.tensor import Tensor

        c_sum(a._c_tensor, out_tensor._c_tensor, axis, keepdim)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        c_sum_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)


class Mean(Function):
    """
    Reduction: mean over a given axis.

    Forward:
        out = mean(a, axis)

    Backward:
        dL/da = broadcast((1/N) * dL/dout, shape of a)

    Notes:
        - Uses C-level kernel `c_mean`.
    """

    lazy_op_class = LazyMean

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", axis: int = 0, keepdim: bool = True
    ) -> "Tensor":
        from py.core.tensor import Tensor

        c_mean(a._c_tensor, out_tensor._c_tensor, axis, keepdim)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        c_mean_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)


class Max(Function):
    """
    Reduction: maximum over a given axis.

    Forward:
        out = max(a, axis)

    Backward:
        dL/da = dL/dout if a is the max element, else 0

    Notes:
        - Uses C-level kernel `c_max`.
        - Backward not yet implemented (TODO: `max_grad_op`).
    """

    lazy_op_class = LazyMax

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", axis: int = 0, keepdim: bool = True
    ) -> "Tensor":
        from py.core.tensor import Tensor

        c_max(a._c_tensor, out_tensor._c_tensor, axis, keepdim)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        c_max_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)
