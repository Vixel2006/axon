from __future__ import annotations
import ctypes
from typing import Any, Tuple, Optional, List
import sys

from py.elnawah_bindings.c_wrapper_functions import (
    c_view,
    c_unsqueeze,
    c_squeeze,
    c_transpose,
    c_expand,
    c_broadcast,
)
from py.elnawah_bindings.ctypes_definitions import CTensor, BackwardFnType
from py.elnawah_bindings.c_library_loader import tensor_lib
from py.ops.lazy_ops.movement_ops import (
    LazyView,
    LazyUnsqueeze,
    LazySqueeze,
    LazyTranspose,
    LazyExpand,
    LazyBroadcast,
)
from py.ops.base import Function


class View(Function):
    """
    Reshape tensor without copying data.

    Forward:
        out = view(a, shape)

    Backward:
        dL/da = reshape(dL/dout, shape of a)

    Notes:
        - Uses C-level kernel `c_view`.
    """

    lazy_op_class = LazyView

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", shape: list
    ) -> "Tensor":
        from py.core.tensor import Tensor

        c_view(a._c_tensor, out_tensor._c_tensor, shape, len(shape))

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        pass


class Unsqueeze(Function):
    """
    Insert a dimension of size 1 at the given axis.

    Forward:
        out = unsqueeze(a, dim)

    Backward:
        dL/da = squeeze(dL/dout, dim)

    Notes:
        - Uses C-level kernel `c_unsqueeze`.
    """

    lazy_op_class = LazyUnsqueeze

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", dim: int = 0
    ) -> "Tensor":
        from py.core.tensor import Tensor

        c_unsqueeze(a._c_tensor, out_tensor._c_tensor, dim)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        pass


class Squeeze(Function):
    """
    Remove a dimension of size 1 at the given axis.

    Forward:
        out = squeeze(a, dim)

    Backward:
        dL/da = unsqueeze(dL/dout, dim)

    Notes:
        - Uses C-level kernel `c_squeeze`.
    """

    lazy_op_class = LazySqueeze

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", dim: int = 0
    ) -> "Tensor":
        from py.core.tensor import Tensor

        c_squeeze(a._c_tensor, out_tensor._c_tensor, dim)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        pass


class Transpose(Function):
    """
    Swap two tensor dimensions.

    Forward:
        out = transpose(a, n, m)

    Backward:
        dL/da = transpose(dL/dout, n, m)

    Notes:
        - Uses C-level kernel `c_transpose`.
    """

    lazy_op_class = LazyTranspose

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", n: int = -2, m: int = -1
    ) -> "Tensor":
        from py.core.tensor import Tensor

        c_transpose(a._c_tensor, out_tensor._c_tensor, n, m)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        pass


class Expand(Function):
    """
    Expand tensor to a larger shape.

    Forward:
        out = expand(a, shape)

    Backward:
        dL/da = reduce(dL/dout, original shape of a)

    Notes:
        - Uses C-level kernel `c_expand`.
    """

    lazy_op_class = LazyExpand

    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return None

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", shape: list[int]
    ) -> "Tensor":
        from py.core.tensor import Tensor

        c_expand(a._c_tensor, out_tensor._c_tensor, shape)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        pass


class Broadcast(Function):
    """
    Broadcast tensor to a larger shape.

    Forward:
        out = broadcast(a, shape)

    Backward:
        dL/da = reduce(dL/dout, original shape of a)

    Notes:
        - Uses C-level kernel `c_broadcast`.
    """

    lazy_op_class = LazyBroadcast

    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return None

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", shape: list[int], ndim: int
    ) -> "Tensor":
        from py.core.tensor import Tensor

        c_broadcast(a._c_tensor, out_tensor._c_tensor, ndim, shape)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        pass
