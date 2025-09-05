from __future__ import annotations
from typing import Any
import ctypes
from .op import LazyOp
from idrak.idrak_bindings.ctypes_definitions import CTensor
from idrak.idrak_bindings.c_wrapper_functions import (
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
    def calc_out_shape(a: "Tensor", dim: int | None, keepdim: bool) -> tuple[int, ...]:
        if dim is None:
            return (1,)
        
        new_shape = list(a.shape)
        if keepdim:
            new_shape[dim] = 1
        else:
            new_shape.pop(dim)
        return tuple(new_shape)


class Sum(ROp):
    @staticmethod
    def forward(out: "Tensor", a: "Tensor", dim: int | None, keepdim: bool) -> "Tensor":
        if dim is None:
            c_sum_full(a._c_tensor, out._c_tensor)
        else:
            if dim < 0:
                dim = a.ndim + dim
            c_sum(a._c_tensor, out._c_tensor, dim, keepdim)
        return out

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras):
        if extras is None: # This means it was a full reduction
            c_sum_full_grad_op(out_ptr, prev_ptrs, n_prev, extras)
        else:
            c_sum_grad_op(out_ptr, prev_ptrs, n_prev, extras)

class Mean(ROp):
    @staticmethod
    def forward(out: "Tensor", a: "Tensor", dim: int | None, keepdim: bool) -> "Tensor":
        if dim is None:
            c_mean_full(a._c_tensor, out._c_tensor)
        else:
            if dim < 0:
                dim = a.ndim + dim
            c_mean(a._c_tensor, out._c_tensor, dim, keepdim)
        return out

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras):
        if extras is None: # This means it was a full reduction
            c_mean_full_grad_op(out_ptr, prev_ptrs, n_prev, extras)
        else:
            c_mean_grad_op(out_ptr, prev_ptrs, n_prev, extras)

class Max(ROp):
    @staticmethod
    def forward(out: "Tensor", a: "Tensor", dim: int | None, keepdim: bool) -> "Tensor":
        if dim is None:
            c_max_full(a._c_tensor, out._c_tensor)
        else:
            if dim < 0:
                dim = a.ndim + dim
            c_max(a._c_tensor, out._c_tensor, dim, keepdim)
        return out

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras):
        if extras is None: # This means it was a full reduction
            c_max_full_grad_op(out_ptr, prev_ptrs, n_prev, extras)
        else:
            c_max_grad_op(out_ptr, prev_ptrs, n_prev, extras)
