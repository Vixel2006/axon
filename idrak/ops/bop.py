from __future__ import annotations
from typing import Any
import ctypes
import math
from .op import LazyOp
from idrak.idrak_bindings.ctypes_definitions import CTensor, Conv2DBackwardExtras
from idrak.idrak_bindings.c_wrapper_functions import (
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
    c_pow_grad_op,
    c_matmul_grad_op,
    c_div_grad_op,
    c_rdiv_grad_op,
    c_rsub_grad_op,
    c_conv_grad_op,
    c_dot,
    c_dot_grad_op
)

class BOp(LazyOp):
    @staticmethod
    def create_ctx_struct(a: "Tensor", b: "Tensor" | float) -> Any:
        if not isinstance(b, CTensor):
            return ctypes.c_float(b)

    @staticmethod
    def compute_broadcasted_shape(shape1: tuple[int, ...], shape2: tuple[int, ...]) -> tuple[int, ...]:
        # Pad the shorter shape with 1s at the beginning
        max_ndim = max(len(shape1), len(shape2))
        padded_shape1 = (1,) * (max_ndim - len(shape1)) + shape1
        padded_shape2 = (1,) * (max_ndim - len(shape2)) + shape2

        result_shape = []
        for dim1, dim2 in zip(padded_shape1, padded_shape2):
            if dim1 == dim2:
                result_shape.append(dim1)
            elif dim1 == 1:
                result_shape.append(dim2)
            elif dim2 == 1:
                result_shape.append(dim1)
            else:
                raise ValueError(f"Shapes are not broadcastable: {shape1} and {shape2}")
        return tuple(result_shape)

    @staticmethod
    def calc_out_shape(a: "Tensor", b: "Tensor"):
        if isinstance(b, CTensor):
            return BOp.compute_broadcasted_shape(a.shape, b.shape)
        return a.shape

class Add(BOp):
    @staticmethod
    def forward(out: "Tensor", a: "Tensor", b: "Tensor" | float) -> "Tensor":
        if isinstance(b, CTensor):
            # Explicitly broadcast a and b to the shape of out
            a = a.broadcast(out.shape)
            b = b.broadcast(out.shape)
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
            # Explicitly broadcast a and b to the shape of out
            a = a.broadcast(out.shape)
            b = b.broadcast(out.shape)
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
            # Explicitly broadcast a and b to the shape of out
            a = a.broadcast(out.shape)
            b = b.broadcast(out.shape)
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
            # Explicitly broadcast a and b to the shape of out
            a = a.broadcast(out.shape)
            b = b.broadcast(out.shape)
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
            print("Still in work")
        else:
            scalar = ctypes.c_float(b)
            c_pow_scalar(a._c_tensor, scalar, out._c_tensor)
        return out
    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras):
        c_pow_grad_op(out_ptr, prev_ptrs, n_prev, extras)

class MatMul(BOp):
    @staticmethod
    def calc_out_shape(a: "Tensor", b: "Tensor"):
        # Get the effective shapes for broadcasting batch dimensions
        a_effective_shape = a.shape[:-2] if a.ndim >= 2 else ()
        b_effective_shape = b.shape[:-2] if b.ndim >= 2 else ()

        # Get the matrix dimensions
        a_K = a.shape[-1]
        b_K = b.shape[-2] if b.ndim >= 2 else b.shape[-1] # If b is 1D, it's K

        if a_K != b_K:
            raise ValueError(f"Matrix multiplication dimensions are incompatible: {a.shape} and {b.shape}")

        # Calculate the broadcasted batch shape
        max_ndim_batch = max(len(a_effective_shape), len(b_effective_shape))
        padded_a_batch_shape = (1,) * (max_ndim_batch - len(a_effective_shape)) + a_effective_shape
        padded_b_batch_shape = (1,) * (max_ndim_batch - len(b_effective_shape)) + b_effective_shape

        result_batch_shape = []
        for dim1, dim2 in zip(padded_a_batch_shape, padded_b_batch_shape):
            if dim1 == dim2:
                result_batch_shape.append(dim1)
            elif dim1 == 1:
                result_batch_shape.append(dim2)
            elif dim2 == 1:
                result_batch_shape.append(dim1)
            else:
                raise ValueError(f"Batch shapes are not broadcastable: {a.shape} and {b.shape}")

        # Determine N and M for the output
        a_N = a.shape[-2] if a.ndim >= 2 else 1
        b_M = b.shape[-1] if b.ndim >= 2 else 1

        # Special handling for 1D inputs resulting in 1D or scalar output
        if a.ndim == 1 and b.ndim == 1:
            return (1,) # Dot product
        elif a.ndim == 1: # a is (K,), b is (..., K, M) -> (..., M)
            return tuple(result_batch_shape) + (b_M,)
        elif b.ndim == 1: # a is (..., N, K), b is (K,) -> (..., N)
            return tuple(result_batch_shape) + (a_N,)
        else: # Both are >= 2D
            return tuple(result_batch_shape) + (a_N, b_M)

    @staticmethod
    def forward(out: "Tensor", a: "Tensor", b: "Tensor" | float) -> "Tensor":
        if isinstance(b, CTensor):
            # Determine the target shapes for broadcasting a and b
            # The batch dimensions of a and b should be broadcasted to the batch dimensions of out
            # The last two dimensions of a and b should remain as they are for matmul
            
            out_batch_shape = out.shape[:-2] if out.ndim >= 2 else ()

            # Construct target shapes for a and b for broadcasting
            # If a or b was originally 1D, we need to promote it to 2D for broadcasting
            a_target_shape = out_batch_shape
            if a.ndim == 1:
                a_target_shape += (1, a.shape[0])
            else:
                a_target_shape += a.shape[-2:]

            b_target_shape = out_batch_shape
            if b.ndim == 1:
                b_target_shape += (b.shape[0], 1)
            else:
                b_target_shape += b.shape[-2:]

            a_broadcasted = a.broadcast(a_target_shape)
            b_broadcasted = b.broadcast(b_target_shape)

            N = a_broadcasted.shape[-2]
            K = a_broadcasted.shape[-1]
            M = b_broadcasted.shape[-1]

            if a_broadcasted.shape[-1] != b_broadcasted.shape[-2]:
                raise RuntimeError(f"Matrix multiplication dimensions are incompatible after broadcasting: {a_broadcasted.shape} and {b_broadcasted.shape}")

            c_matmul(a_broadcasted._c_tensor, b_broadcasted._c_tensor, out._c_tensor, N=N, K=K, P=M)
        else:
            raise TypeError("MatMul does not support scalar multiplication.")
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
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras): c_conv_grad_op(out_ptr, prev_ptrs, n_prev, extras)

class Dot(BOp):
    @staticmethod
    def calc_out_shape(a: "Tensor", b: "Tensor"):
        return (1,)

    @staticmethod
    def forward(out: "Tensor", a: "Tensor", b: "Tensor") -> "Tensor":
        c_dot(a._c_tensor, b._c_tensor, out._c_tensor)
        return out

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras):
        c_dot_grad_op(out_ptr, prev_ptrs, n_prev, extras)
