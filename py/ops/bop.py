from __future__ import annotations
from typing import Any
import ctypes
import math
from .op import LazyOp
from py.elnawah_bindings.ctypes_definitions import CTensor, Conv2DBackwardExtras
from py.elnawah_bindings.c_wrapper_functions import (
    c_add,
    c_sub,
    c_mul,
    c_matmul,
    c_div,
    c_pow_scalar,
    c_div_scalar,
    c_add_scalar,
    c_sub_scalar,
    c_rsub_scalar,
    c_mul_scalar,
    c_conv,
    c_rdiv_scalar,
    c_rdiv_scalar,
    c_add_grad_op,
    c_sub_grad_op,
    c_mul_grad_op,
    c_matmul_grad_op,
    c_div_grad_op,
    c_rdiv_grad_op,
    c_rsub_grad_op,
    c_conv_grad_op
)

class BOp(LazyOp):
    @staticmethod
    def create_ctx_struct(a: "Tensor", b: "Tensor" | float) -> Any:
        if not isinstance(b, CTensor):
            return ctypes.c_float(b)
    @staticmethod
    def calc_out_shape(a: "Tensor", b: "Tensor"): return a.shape

class Add(BOp):
    @staticmethod
    def forward(out: "Tensor", a: "Tensor", b: "Tensor" | float) -> "Tensor":
        if isinstance(b, CTensor):
            c_add(a._c_tensor, b._c_tensor, out._c_tensor)
        else:
            scalar = ctypes.c_float(b)
            c_add_scalar(a._c_tensor, scalar, out._c_tensor)
        return out

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras): c_add_grad_op(out_ptr, prev_ptrs, n_prev, extras)

class Sub(BOp):
    @staticmethod
    def forward(out: "Tensor", a: "Tensor" | float, b: "Tensor" | float) -> "Tensor": 
        if isinstance(b, CTensor):
            c_sub(a._c_tensor, b._c_tensor, out._c_tensor)
        else:
            scalar = ctypes.c_float(b)
            c_sub_scalar(a._c_tensor, scalar, out._c_tensor)
        return out

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras): c_sub_grad_op(out_ptr, prev_ptrs, n_prev, extras)

class RSub(BOp):
    @staticmethod
    def forward(out: "Tensor", a: "Tensor", b: float) -> "Tensor": 
        scalar = ctypes.c_float(b)
        c_rsub_scalar(scalar, a._c_tensor, out._c_tensor)
        return out

    @staticmethod
    def backward(out_ptr: ctypes.pointer(ctensor), prev_ptrs: ctypes.pointer(ctypes.pointer(ctensor)), n_prev: int, extras): c_rsub_grad_op(out_ptr, prev_ptrs, n_prev, extras)

class Mul(BOp):
    @staticmethod
    def forward(out: "Tensor", a: "Tensor", b: "Tensor" | float) -> "Tensor":
        if isinstance(b, CTensor):
            c_mul(a._c_tensor, b._c_tensor, out._c_tensor)
        else:
            scalar = ctypes.c_float(b)
            c_mul_scalar(a._c_tensor, scalar, out._c_tensor)
        return out

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras): c_mul_grad_op(out_ptr, prev_ptrs, n_prev, extras)

class Div(BOp):
    @staticmethod
    def forward(out: "Tensor", a: "Tensor", b: "Tensor" | float) -> "Tensor":
        if isinstance(b, CTensor):
            c_div(a._c_tensor, b._c_tensor, out._c_tensor)
        else:
            scalar = ctypes.c_float(b)
            c_div_scalar(a._c_tensor, scalar, out._c_tensor)
        return out

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras): c_div_grad_op(out_ptr, prev_ptrs, n_prev, extras)

class RDiv(BOp):
    @staticmethod
    def forward(out: "Tensor", a: "Tensor", b: "Tensor" | float) -> "Tensor":
        scalar = ctypes.c_float(b)
        c_rdiv_scalar(scalar, a._c_tensor, out._c_tensor)
        return out

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras):
        c_rdiv_grad_op(out_ptr, prev_ptrs, n_prev, extras)

class Pow(BOp):
    @staticmethod
    def forward(out: "Tensor", a: "Tensor", b: "Tensor" | float) -> "Tensor":
        if isinstance(b, CTensor):
            print("Still in work")#c_mul(a._c_tensor, b._c_tensor, out._c_tensor)
        else:
            scalar = ctypes.c_float(b)
            c_pow_scalar(a._c_tensor, scalar, out._c_tensor)
        return out

class MatMul(BOp):
    @staticmethod
    def forward(out: "Tensor", a: "Tensor", b: "Tensor" | float) -> "Tensor":
        N = a.shape[-2]
        K = a.shape[-1]
        M = b.shape[-1]

        if K != b.shape[-2]:
            raise RuntimeError("Can't do this shit")
        c_matmul(a._c_tensor, b._c_tensor, out._c_tensor, N=N, K=K, P=M)

        return out

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras):
        c_matmul_grad_op(out_ptr, prev_ptrs, n_prev, extras)


class Conv2D(BOp):
    @staticmethod
    def create_ctx_struct(a: "Tensor", b: "Tensor", kernel_size: tuple[int, ...], stride: tuple[int, int], padding: int) -> Any:
        Hin = a.shape[2]
        Win = a.shape[3]

        Kh = b.shape[1]
        Kw = b.shape[2]

        Sh = stride[0]
        Sw = stride[1]

        Hout = math.floor((Hin - Kh + 2 * padding + 1) / Sh)
        Wout = math.floor((Win - Kw + 2 * padding + 1) / Sw)

        ctx = Conv2DBackwardExtras(
            padding=padding,
            H_in=Hin,
            W_in=Win,
            Kh=Kh,
            Kw=Kw,
            Sh=Sh,
            Sw=Sw,
            Hout=Hout,
            Wout=Wout,
        )

        ctx = ctypes.pointer(ctx)

        return ctx

    @staticmethod
    def calc_out_shape(
        a: "Tensor", b: "Tensor", kernel_size: tuple[int, ...], stride: tuple[int, int], padding: int
    ):
        Cout = b.shape[1]

        Hin = a.shape[2]
        Win = a.shape[3]

        Kh = b.shape[1]
        Kw = b.shape[2]

        Hout = math.floor((Hin - Kh + 2 * padding + 1) / stride[0])
        Wout = math.floor((Win - Kw + 2 * padding + 1) / stride[1])

        return (a.shape[1], Cout, Hout, Wout)

    @staticmethod
    def forward(
        out: "Tensor", a: "Tensor", b: "Tensor", kernel_size: tuple[int, ...], stride: tuple[int, int], padding: int
        ) -> "Tensor":
        c_conv(a._c_tensor, b._c_tensor, out._c_tensor, kernel_size, stride, padding)
        return out
    
    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras):
        c_conv_grad_op(out_ptr, prev_ptrs, n_prev, extras)
