from __future__ import annotations
import ctypes
from typing import Any, Tuple, Optional, List
import sys

from py.elnawah_bindings.c_wrapper_functions import (
    c_add,
    c_sub,
    c_mul,
    c_div,
    c_matmul,
    c_add_scalar,
    c_sub_scalar,
    c_rsub_scalar,
    c_mul_scalar,
    c_div_scalar,
    c_rdiv_scalar,
    c_matmul_grad_op,
    c_add_grad_op,
    c_sub_grad_op,
    c_rsub_grad_op,
    c_mul_grad_op,
    c_div_grad_op,
    c_rdiv_grad_op,
)
from py.elnawah_bindings.ctypes_definitions import CTensor, BackwardFnType
from py.elnawah_bindings.c_library_loader import tensor_lib
from py.ops.lazy_ops.binary_ops import (
    LazyAdd,
    LazySub,
    LazyRSub,
    LazyMul,
    LazyDiv,
    LazyRDiv,
    LazyMatMul,
)
from py.ops.base import Function


class Add(Function):
    """
    Elementwise addition.

    Forward:
        out = a + b

    Backward:
        dL/da = dL/dout
        dL/db = dL/dout

    Notes:
        - Supports tensor-tensor and tensor-scalar addition.
        - Registers C-level backward function `c_add_grad_op`.
    """

    lazy_op_class = LazyAdd

    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.add_grad_op)

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", b: "Tensor"
    ) -> "Tensor":
        from py.core.tensor import Tensor

        if isinstance(b, (Tensor, CTensor)):
            c_add(a._c_tensor, b._c_tensor, out_tensor._c_tensor)
        else:
            scalar_val = ctypes.c_float(b)
            self.extras = scalar_val
            c_add_scalar(a._c_tensor, b, out_tensor._c_tensor)
        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        c_add_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)


class Sub(Function):
    """
    Elementwise substration.

    Forward:
        out = a - b

    Backward:
        dL/da = dL/dout
        dL/db = -dL/dout

    Notes:
        - Supports tensor-tensor and tensor-scalar addition.
        - Registers C-level backward function `c_sub_grad_op`.
    """

    lazy_op_class = LazySub

    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.sub_grad_op)

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", b: "Tensor"
    ) -> "Tensor":
        from py.core.tensor import Tensor

        if isinstance(b, (Tensor, CTensor)):
            c_sub(a._c_tensor, b._c_tensor, out_tensor._c_tensor)
        else:
            scalar_val = ctypes.c_float(b)
            self.extras = scalar_val
            c_sub_scalar(a._c_tensor, b, out_tensor._c_tensor)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        c_sub_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)


class RSub(Function):
    """
    Reverse elementwise substraction (scalar - tensor).

    Forward:
        out = b - a

    Backward:
        dL/da = -dL/dout

    Notes:
        - Only supports scalar substracted by tensor.
        - Uses C-level kernel `c_rsub_scalar`.
        - Registers C-level backward function `c_rsub_grad_op`.
    """

    lazy_op_class = LazyRSub

    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.rsub_grad_op)

    def _execute_forward(self, out_tensor: "Tensor", a: "Tensor", b: float) -> "Tensor":
        from py.core.tensor import Tensor

        scalar_val = ctypes.c_float(b)
        self.extras = scalar_val
        c_rsub_scalar(b, a._c_tensor, out_tensor._c_tensor)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        c_rsub_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)


class Mul(Function):
    """
    Elementwise multiplication.

    Forward:
        out = a * b

    Backward:
        dL/da = b * dL/dout
        dL/db = a * dL/dout

    Notes:
        - Supports tensor-tensor and tensor-scalar multiplication.
        - Uses C-level kernels `c_mul` and `c_mul_scalar`.
        - Registers C-level backward function `c_mul_grad_op`.
    """

    lazy_op_class = LazyMul

    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.mul_grad_op)

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", b: "Tensor"
    ) -> "Tensor":
        from py.core.tensor import Tensor

        if isinstance(b, (Tensor, CTensor)):
            c_mul(a._c_tensor, b._c_tensor, out_tensor._c_tensor)
        else:  # scalar
            scalar_val = ctypes.c_float(b)
            self.extras = scalar_val
            c_mul_scalar(a._c_tensor, b, out_tensor._c_tensor)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        c_mul_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)


class Div(Function):
    """
    Elementwise division.

    Forward:
        out = a / b

    Backward:
        dL/da = (1 / b) * dL/dout
        dL/db = -(a / b^2) * dL/dout

    Notes:
        - Supports tensor-tensor and tensor-scalar division.
        - Uses C-level kernels `c_div` and `c_div_scalar`.
        - Registers C-level backward function `c_div_grad_op`.
    """

    lazy_op_class = LazyDiv

    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.div_grad_op)

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", b: "Tensor"
    ) -> "Tensor":
        from py.core.tensor import Tensor

        if isinstance(b, (Tensor, CTensor)):
            c_div(a._c_tensor, b._c_tensor, out_tensor._c_tensor)
        else:  # scalar
            scalar_val = ctypes.c_float(b)
            self.extras = scalar_val
            c_div_scalar(a._c_tensor, b, out_tensor._c_tensor)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        c_div_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)


class RDiv(Function):
    """
    Reverse elementwise division (scalar / tensor).

    Forward:
        out = b / a

    Backward:
        dL/da = -(b / a^2) * dL/dout

    Notes:
        - Only supports scalar divided by tensor.
        - Uses C-level kernel `c_rdiv_scalar`.
        - Registers C-level backward function `c_rdiv_grad_op`.
    """

    lazy_op_class = LazyRDiv

    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.rdiv_grad_op)

    def _execute_forward(self, out_tensor: "Tensor", a: "Tensor", b: float) -> "Tensor":
        from py.core.tensor import Tensor

        scalar_val = ctypes.c_float(b)
        self.extras = scalar_val
        c_rdiv_scalar(b, a._c_tensor, out_tensor._c_tensor)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        c_rdiv_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)


class MatMul(Function):
    """
    Matrix multiplication.

    Forward:
        out = a @ b

    Backward:
        dL/da = dL/dout @ b^T
        dL/db = a^T @ dL/dout

    Notes:
        - Requires that a.shape[-1] == b.shape[-2].
        - Uses C-level kernel `c_matmul`.
    """

    lazy_op_class = LazyMatMul

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", b: "Tensor"
    ) -> "Tensor":
        from py.core.tensor import Tensor

        if a.shape[-1] != b.shape[-2]:
            raise RuntimeError(
                f"Can't multiply tensors with shapes {a.shape} and {b.shape}"
            )

        N = a.shape[-2]
        K = a.shape[-1]
        P = b.shape[-1]

        c_matmul(a._c_tensor, b._c_tensor, out_tensor._c_tensor, N, K, P)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        c_matmul_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)
