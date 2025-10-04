from __future__ import annotations
from typing import Any
import ctypes
from .op import LazyOp
from axon.axon_bindings.ctypes_definitions import CTensor
from axon.axon_bindings.c_wrapper_functions import (
    get_op_function
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
            get_op_function("sum_full", out.device)(a.c_tensor_ptr, out.c_tensor_ptr)
        else:
            if dim < 0:
                dim = a.ndim + dim
            get_op_function("sum", out.device)(a.c_tensor_ptr, out.c_tensor_ptr, dim, keepdim)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        if extras is None:
            get_op_function("sum_full_grad", out_ptr.contents.device)(out_ptr, prev_ptrs, n_prev, extras)
        else:
            get_op_function("sum_grad", out_ptr.contents.device)(out_ptr, prev_ptrs, n_prev, extras)

class Mean(ROp):
    @staticmethod
    def forward(out: "Tensor", a: "Tensor", *, dim: int | None, keepdim: bool):
        if dim is None:
            get_op_function("mean_full", out.device)(a.c_tensor_ptr, out.c_tensor_ptr)
        else:
            if dim < 0:
                dim = a.ndim + dim
            get_op_function("mean", out.device)(a.c_tensor_ptr, out.c_tensor_ptr, dim, keepdim)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        if extras is None:
            get_op_function("mean_full_grad", out_ptr.contents.device)(out_ptr, prev_ptrs, n_prev, extras)
        else:
            get_op_function("mean_grad", out_ptr.contents.device)(out_ptr, prev_ptrs, n_prev, extras)

class Max(ROp):
    @staticmethod
    def forward(out: "Tensor", a: "Tensor", *, dim: int | None, keepdim: bool):
        if dim is None:
            get_op_function("max_full", out.device)(a.c_tensor_ptr, out.c_tensor_ptr)
        else:
            if dim < 0:
                dim = a.ndim + dim
            get_op_function("max", out.device)(a.c_tensor_ptr, out.c_tensor_ptr, dim, keepdim)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        if extras is None:
            get_op_function("max_full_grad", out_ptr.contents.device)(out_ptr, prev_ptrs, n_prev, extras)
        else:
            get_op_function("max_grad", out_ptr.contents.device)(out_ptr, prev_ptrs, n_prev, extras)
