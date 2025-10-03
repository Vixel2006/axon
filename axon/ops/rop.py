from __future__ import annotations
from typing import Any
import ctypes
from .op import LazyOp
from axon.axon_bindings.ctypes_definitions import CTensor
from axon.axon_bindings.c_wrapper_functions import (
    c_sum,
    c_mean,
    c_max,
    c_sum_full,
    c_mean_full,
    c_max_full,
    c_sum_grad_op,
    c_mean_grad_op,
    c_max_grad_op,
    c_sum_full_grad_op,
    c_mean_full_grad_op,
    c_max_full_grad_op
)

class ROp(LazyOp):
    @staticmethod
    def calc_out_shape(*args: Any, **kwargs: Any) -> tuple[int, ...]:
        from axon.core.tensor import Tensor
        
        a_tensor: Optional[Tensor] = None
        if args and isinstance(args[0], Tensor):
            a_tensor = args[0]
        else:
            raise TypeError("First argument to reduction operation must be a Tensor for calc_out_shape.")

        dim = kwargs.get('dim', None)
        keepdim = kwargs.get('keepdim', False)

        if dim is None:
            return (1,)
        
        new_shape = list(a_tensor.shape)
        if dim < 0:
            dim = a_tensor.ndim + dim

        if keepdim:
            new_shape[dim] = 1
        else:
            new_shape.pop(dim)
        return tuple(new_shape)

    @staticmethod
    def create_ctx_struct(*args: Any, **kwargs: Any) -> Tuple[Dict[str, Any], Any]:
        from axon.core.tensor import Tensor

        a_tensor: Optional[Tensor] = None
        if args and isinstance(args[0], Tensor):
            a_tensor = args[0]
        else:
            raise TypeError("First argument to reduction operation must be a Tensor.")

        dim = kwargs.get('dim', None)
        keepdim = kwargs.get('keepdim', False)

        forward_kwargs: Dict[str, Any] = {
            'dim': dim,
            'keepdim': keepdim
        }
        
        backward_ctx: Any = None
        if dim is not None:
            backward_ctx = ctypes.c_int(0)
        
        return forward_kwargs, None


class Sum(ROp):
    @staticmethod
    def forward(out: "Tensor", a: "Tensor", *, dim: int | None, keepdim: bool):
        if dim is None:
            c_sum_full(a.c_tensor_ptr, out.c_tensor_ptr)
        else:
            if dim < 0:
                dim = a.ndim + dim
            c_sum(a.c_tensor_ptr, out.c_tensor_ptr, dim, keepdim)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        if extras is None:
            c_sum_full_grad_op(out_ptr, prev_ptrs, n_prev, extras)
        else:
            c_sum_grad_op(out_ptr, prev_ptrs, n_prev, extras)

class Mean(ROp):
    @staticmethod
    def forward(out: "Tensor", a: "Tensor", *, dim: int | None, keepdim: bool):
        if dim is None:
            c_mean_full(a.c_tensor_ptr, out.c_tensor_ptr)
        else:
            if dim < 0:
                dim = a.ndim + dim
            c_mean(a.c_tensor_ptr, out.c_tensor_ptr, dim, keepdim)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        if extras is None:
            c_mean_full_grad_op(out_ptr, prev_ptrs, n_prev, extras)
        else:
            c_mean_grad_op(out_ptr, prev_ptrs, n_prev, extras)

class Max(ROp):
    @staticmethod
    def forward(out: "Tensor", a: "Tensor", *, dim: int | None, keepdim: bool):
        if dim is None:
            c_max_full(a.c_tensor_ptr, out.c_tensor_ptr)
        else:
            if dim < 0:
                dim = a.ndim + dim
            c_max(a.c_tensor_ptr, out.c_tensor_ptr, dim, keepdim)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        if extras is None:
            c_max_full_grad_op(out_ptr, prev_ptrs, n_prev, extras)
        else:
            c_max_grad_op(out_ptr, prev_ptrs, n_prev, extras)
