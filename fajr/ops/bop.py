from __future__ import annotations
from typing import Any
import ctypes
import math
from .op import LazyOp
from fajr.fajr_bindings.ctypes_definitions import CTensor, Conv2DBackwardExtras
from fajr.fajr_bindings.c_wrapper_functions import (
    c_add,
    c_sub,
    c_mul,
    c_matmul,
    c_div,
    c_pow_scalar,
    c_pow,
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
    def create_ctx_struct(*args: Any, **kwargs: Any) -> Tuple[Dict[str, Any], Any]:
        from fajr.core.tensor import Tensor
        a_operand: Any = args[0]

        b_operand: Any = None
        if len(args) > 1:
            b_operand = args[1]
        elif 'scalar_val' in kwargs:
            b_operand = kwargs['scalar_val']

        forward_kwargs: Dict[str, Any] = {}
        backward_ctx: Any = None

        if isinstance(b_operand, (float, int)):
            forward_kwargs["scalar_val"] = float(b_operand)
            backward_ctx = ctypes.cast(ctypes.pointer(ctypes.c_float(float(b_operand))), ctypes.c_void_p)
        elif isinstance(a_operand, (float, int)) and isinstance(b_operand, Tensor):
            forward_kwargs["r_scalar_val"] = float(a_operand)
            backward_ctx = ctypes.cast(ctypes.pointer(ctypes.c_float(float(a_operand))), ctypes.c_void_p)

        return forward_kwargs, backward_ctx

    @staticmethod
    def compute_broadcasted_shape(shape1: tuple[int, ...], shape2: tuple[int, ...]) -> tuple[int, ...]:
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
    def calc_out_shape(*args: Any, **kwargs: Any) -> tuple[int, ...]:
        from fajr.core.tensor import Tensor
        if not args:
            raise ValueError("calc_out_shape requires at least 'a' operand.")

        a_operand: Any = args[0]
        b_operand: Any = None
        if len(args) > 1:
            b_operand = args[1]
        elif 'scalar_val' in kwargs:
            b_operand = kwargs['scalar_val']
        elif 'r_scalar_val' in kwargs:
            b_operand = kwargs['r_scalar_val']

        if isinstance(a_operand, (Tensor, CTensor)):
            if isinstance(b_operand, (Tensor, CTensor)):
                b_tensor_shape = b_operand.shape
                return BOp.compute_broadcasted_shape(a_operand.shape, b_tensor_shape)
            elif isinstance(b_operand, (float, int)):
                return a_operand.shape
            else:
                raise ValueError(f"Second operand for BOp (Tensor, Any) must be a Tensor or a scalar, got {type(b_operand)} (or None).")
        elif isinstance(a_operand, (float, int)): # Handles reverse scalar ops where scalar is first arg
            if isinstance(b_operand, Tensor):
                return b_operand.shape
            else:
                raise ValueError(f"Second operand for BOp (scalar, Any) must be a Tensor, got {type(b_operand)} (or None).")
        else:
            raise TypeError(f"First operand for BOp must be a Tensor or a scalar, got {type(a_operand)}")


class Add(BOp):
    @staticmethod
    def calc_out_shape(*args: Any, **kwargs: Any) -> tuple[int, ...]:
        return BOp.calc_out_shape(*args, **kwargs)

    @staticmethod
    def forward(out: "Tensor", a_tensor: "Tensor", b_tensor: Optional["Tensor"] = None, scalar_val: Optional[float] = None):
        if b_tensor is not None:
            a_broadcasted = a_tensor.broadcast(out.shape).realize()
            b_broadcasted = b_tensor.broadcast(out.shape).realize()
            c_add(a_broadcasted.c_tensor_ptr, b_broadcasted.c_tensor_ptr, out.c_tensor_ptr)
        elif scalar_val is not None:
            a_realized = a_tensor.realize()
            scalar = ctypes.c_float(scalar_val)
            c_add_scalar(a_realized.c_tensor_ptr, scalar, out.c_tensor_ptr)
        else:
            raise ValueError("Add operation requires either a Tensor or a scalar for its second operand.")

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        c_add_grad_op(out_ptr, prev_ptrs, n_prev, extras)


class Sub(BOp):
    @staticmethod
    def calc_out_shape(*args: Any, **kwargs: Any) -> tuple[int, ...]:
        return BOp.calc_out_shape(*args, **kwargs)

    @staticmethod
    def forward(out: "Tensor", a_tensor: "Tensor", b_tensor: Optional["Tensor"] = None, scalar_val: Optional[float] = None):
        if b_tensor is not None:
            a_broadcasted = a_tensor.broadcast(out.shape).realize()
            b_broadcasted = b_tensor.broadcast(out.shape).realize()
            c_sub(a_broadcasted.c_tensor_ptr, b_broadcasted.c_tensor_ptr, out.c_tensor_ptr)
        elif scalar_val is not None:
            a_realized = a_tensor.realize()
            scalar = ctypes.c_float(scalar_val)
            c_sub_scalar(a_realized.c_tensor_ptr, scalar, out.c_tensor_ptr)
        else:
            raise ValueError("Sub operation requires either a Tensor or a scalar for its second operand.")

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        c_sub_grad_op(out_ptr, prev_ptrs, n_prev, extras)


class RSub(BOp):
    @staticmethod
    def create_ctx_struct(*args: Any, **kwargs: Any) -> Tuple[Dict[str, Any], Any]:
        # In RSub, args[0] is the scalar, args[1] is the Tensor
        if len(args) < 1 or not isinstance(args[0], (float, int)):
            raise TypeError("RSub (scalar - Tensor) operation expected first operand to be a scalar.")

        forward_kwargs: Dict[str, Any] = {"r_scalar_val": float(args[0])}
        backward_ctx: Any = ctypes.cast(ctypes.pointer(ctypes.c_float(float(args[0]))), ctypes.c_void_p)

        return forward_kwargs, backward_ctx

    @staticmethod
    def calc_out_shape(*args: Any, **kwargs: Any) -> tuple[int, ...]:
        return BOp.calc_out_shape(*args, **kwargs)

    @staticmethod
    def forward(out: "Tensor", b_tensor: "Tensor", r_scalar_val: float):
        b_realized = b_tensor.realize()
        scalar = ctypes.c_float(r_scalar_val)
        c_rsub_scalar(scalar, b_realized.c_tensor_ptr, out.c_tensor_ptr)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        c_rsub_grad_op(out_ptr, prev_ptrs, n_prev, extras)


class Mul(BOp):
    @staticmethod
    def calc_out_shape(*args: Any, **kwargs: Any) -> tuple[int, ...]:
        return BOp.calc_out_shape(*args, **kwargs)

    @staticmethod
    def forward(out: "Tensor", a_tensor: "Tensor", b_tensor: Optional["Tensor"] = None, scalar_val: Optional[float] = None):
        if b_tensor is not None:
            a_broadcasted = a_tensor.broadcast(out.shape).realize()
            b_broadcasted = b_tensor.broadcast(out.shape).realize()
            c_mul(a_broadcasted.c_tensor_ptr, b_broadcasted.c_tensor_ptr, out.c_tensor_ptr)
        elif scalar_val is not None:
            a_realized = a_tensor.realize()
            scalar = ctypes.c_float(scalar_val)
            c_mul_scalar(a_realized.c_tensor_ptr, scalar, out.c_tensor_ptr)
        else:
            raise ValueError("Mul operation requires either a Tensor or a scalar for its second operand.")

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        c_mul_grad_op(out_ptr, prev_ptrs, n_prev, extras)


class Div(BOp):
    @staticmethod
    def calc_out_shape(*args: Any, **kwargs: Any) -> tuple[int, ...]:
        return BOp.calc_out_shape(*args, **kwargs)

    @staticmethod
    def forward(out: "Tensor", a_tensor: "Tensor", b_tensor: Optional["Tensor"] = None, scalar_val: Optional[float] = None):
        if b_tensor is not None:
            a_broadcasted = a_tensor.broadcast(out.shape).realize()
            b_broadcasted = b_tensor.broadcast(out.shape).realize()
            c_div(a_broadcasted.c_tensor_ptr, b_broadcasted.c_tensor_ptr, out.c_tensor_ptr)
        elif scalar_val is not None:
            a_realized = a_tensor.realize()
            scalar = ctypes.c_float(scalar_val)
            c_div_scalar(a_realized.c_tensor_ptr, scalar, out.c_tensor_ptr)
        else:
            raise ValueError("Div operation requires either a Tensor or a scalar for its second operand.")

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        c_div_grad_op(out_ptr, prev_ptrs, n_prev, extras)


class RDiv(BOp):
    @staticmethod
    def create_ctx_struct(*args: Any, **kwargs: Any) -> Tuple[Dict[str, Any], Any]:
        if len(args) < 1 or not isinstance(args[0], (float, int)):
            raise TypeError("RDiv (scalar / Tensor) operation expected first operand to be a scalar.")

        forward_kwargs: Dict[str, Any] = {"r_scalar_val": float(args[0])}
        backward_ctx: Any = ctypes.cast(ctypes.pointer(ctypes.c_float(float(args[0]))), ctypes.c_void_p)

        return forward_kwargs, backward_ctx

    @staticmethod
    def calc_out_shape(*args: Any, **kwargs: Any) -> tuple[int, ...]:
        return BOp.calc_out_shape(*args, **kwargs)

    @staticmethod
    def forward(out: "Tensor", b_tensor: "Tensor", r_scalar_val: float):
        b_realized = b_tensor.realize()
        scalar = ctypes.c_float(r_scalar_val)
        c_rdiv_scalar(scalar, b_realized.c_tensor_ptr, out.c_tensor_ptr)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        c_rdiv_grad_op(out_ptr, prev_ptrs, n_prev, extras)


class Pow(BOp):
    @staticmethod
    def calc_out_shape(*args: Any, **kwargs: Any) -> tuple[int, ...]:
        return BOp.calc_out_shape(*args, **kwargs)

    @staticmethod
    def forward(out: "Tensor", a_tensor: "Tensor", b_tensor: Optional["Tensor"] = None, scalar_val: Optional[float] = None):
        if b_tensor is not None:
            a_broadcasted = a_tensor.broadcast(out.shape).realize()
            b_broadcasted = b_tensor.broadcast(out.shape).realize()
            c_pow(a_broadcasted.c_tensor_ptr, b_broadcasted.c_tensor_ptr, out.c_tensor_ptr)
        elif scalar_val is not None:
            a_realized = a_tensor.realize()
            scalar = ctypes.c_float(scalar_val)
            c_pow_scalar(a_realized.c_tensor_ptr, scalar, out.c_tensor_ptr)
        else:
            raise ValueError("Pow operation requires either a Tensor or a scalar for its second operand.")

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        c_pow_grad_op(out_ptr, prev_ptrs, n_prev, extras)


class MatMul(BOp):
    @staticmethod
    def calc_out_shape(*args: Any, **kwargs: Any) -> tuple[int, ...]:
        from fajr.core.tensor import Tensor
        if len(args) < 2:
            raise ValueError("MatMul.calc_out_shape requires two Tensor operands.")
        a_tensor: Tensor = args[0]
        b_tensor: Tensor = args[1]

        if not isinstance(a_tensor, Tensor) or not isinstance(b_tensor, Tensor):
            raise TypeError("MatMul operands must be Tensors.")

        a_effective_shape = a_tensor.shape[:-2] if a_tensor.ndim >= 2 else ()
        b_effective_shape = b_tensor.shape[:-2] if b_tensor.ndim >= 2 else ()

        a_K = a_tensor.shape[-1]
        b_K = b_tensor.shape[-2] if b_tensor.ndim >= 2 else b_tensor.shape[-1] # For vector * matrix, last dim is treated as K

        if a_K != b_K:
            raise ValueError(f"Matrix multiplication dimensions are incompatible: {a_tensor.shape} and {b_tensor.shape}")

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
                raise ValueError(f"Batch shapes are not broadcastable: {a_tensor.shape} and {b_tensor.shape}")

        a_N = a_tensor.shape[-2] if a_tensor.ndim >= 2 else 1
        b_M = b_tensor.shape[-1] if b_tensor.ndim >= 2 else 1

        if a_tensor.ndim == 1 and b_tensor.ndim == 1:
            return (1,) # dot product, scalar output
        elif a_tensor.ndim == 1: # vector * matrix
            return tuple(result_batch_shape) + (b_M,)
        elif b_tensor.ndim == 1: # matrix * vector
            return tuple(result_batch_shape) + (a_N,)
        else: # matrix * matrix
            return tuple(result_batch_shape) + (a_N, b_M)

    @staticmethod
    def create_ctx_struct(*args: Any, **kwargs: Any) -> Tuple[Dict[str, Any], Any]:
        return {}, None

    @staticmethod
    def forward(out: "Tensor", a_tensor: "Tensor", b_tensor: "Tensor"):
        # Determine effective N, K, M for the C function based on possibly broadcasted shapes
        # and handle 1D tensors correctly.
        N_final: int
        K_final: int
        M_final: int

        if a_tensor.ndim == 1 and b_tensor.ndim == 1:
            N_final = 1
            K_final = a_tensor.shape[0]
            M_final = 1
        elif a_tensor.ndim == 1: # vector @ matrix
            # a_tensor (K,) needs to be treated as (1, K) for matmul
            # out.shape is (..., M)
            N_final = 1
            K_final = a_tensor.shape[0]
            M_final = b_tensor.shape[-1]
        elif b_tensor.ndim == 1: # matrix @ vector
            # b_tensor (K,) needs to be treated as (K, 1) for matmul
            # out.shape is (..., N)
            N_final = a_tensor.shape[-2]
            K_final = a_tensor.shape[-1]
            M_final = 1
        else: # matrix @ matrix
            N_final = a_tensor.shape[-2]
            K_final = a_tensor.shape[-1]
            M_final = b_tensor.shape[-1]

        # Explicit broadcasting must happen if c_matmul does not handle it internally
        # (It's generally better for the Python wrapper to prepare broadcasted inputs)
        a_broadcasted = a_tensor.broadcast(out.shape[:-2] + (N_final, K_final)).realize() if a_tensor.ndim >= 2 else a_tensor.broadcast(out.shape[:-1] + (K_final,)).realize()
        b_broadcasted = b_tensor.broadcast(out.shape[:-2] + (K_final, M_final)).realize() if b_tensor.ndim >= 2 else b_tensor.broadcast(out.shape[:-1] + (K_final,)).realize()

        if a_broadcasted.shape[-1] != (b_broadcasted.shape[-2] if b_broadcasted.ndim >= 2 else b_broadcasted.shape[-1]):
             raise RuntimeError(f"Internal error: Matmul broadcasted dimensions incompatible: {a_broadcasted.shape} and {b_broadcasted.shape}")

        c_matmul(a_broadcasted.c_tensor_ptr, b_broadcasted.c_tensor_ptr, out.c_tensor_ptr, N=N_final, K=K_final, P=M_final)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        c_matmul_grad_op(out_ptr, prev_ptrs, n_prev, extras)


class Conv2D(BOp):
    @staticmethod
    def create_ctx_struct(*args: Any, **kwargs: Any) -> Tuple[Dict[str, Any], Any]:
        from fajr.core.tensor import Tensor
        if len(args) < 2:
            raise ValueError("Conv2D.create_ctx_struct requires at least two Tensor operands (input, kernel).")

        a_tensor: Tensor = args[0] # Input
        b_tensor: Tensor = args[1] # Kernel

        kernel_size = kwargs.get("kernel_size")
        stride = kwargs.get("stride")
        padding = kwargs.get("padding")

        if kernel_size is None or stride is None or padding is None:
            raise ValueError("Conv2D.create_ctx_struct requires kernel_size, stride, and padding as keyword arguments.")
        if not (isinstance(kernel_size, (tuple, list)) and len(kernel_size) == 2):
            raise TypeError("kernel_size must be a tuple/list of 2 integers.")
        if not (isinstance(stride, (tuple, list)) and len(stride) == 2):
            raise TypeError("stride must be a tuple/list of 2 integers.")


        Hin = a_tensor.shape[2]
        Win = a_tensor.shape[3]

        Kh = kernel_size[0]
        Kw = kernel_size[1]

        Sh = stride[0]
        Sw = stride[1]

        # PyTorch-like formula for output size: floor((in_dim + 2*pad - kernel_dim) / stride + 1)
        Hout = math.floor((Hin - Kh + 2 * padding) / Sh + 1)
        Wout = math.floor((Win - Kw + 2 * padding) / Sw + 1)

        backward_ctx_struct = Conv2DBackwardExtras(
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

        forward_kwargs = {
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding
        }

        return forward_kwargs, ctypes.pointer(backward_ctx_struct)

    @staticmethod
    def calc_out_shape(*args: Any, **kwargs: Any) -> tuple[int, ...]:
        from fajr.core.tensor import Tensor
        if len(args) < 2:
            raise ValueError("Conv2D.calc_out_shape requires at least two Tensor operands (input, kernel).")
        a_tensor: Tensor = args[0] # input
        b_tensor: Tensor = args[1] # kernel

        kernel_size = kwargs.get("kernel_size")
        stride = kwargs.get("stride")
        padding = kwargs.get("padding")

        if kernel_size is None or stride is None or padding is None:
            raise ValueError("Conv2D.calc_out_shape requires kernel_size, stride, and padding as keyword arguments.")
        if not (isinstance(kernel_size, (tuple, list)) and len(kernel_size) == 2):
            raise TypeError("kernel_size must be a tuple/list of 2 integers.")
        if not (isinstance(stride, (tuple, list)) and len(stride) == 2):
            raise TypeError("stride must be a tuple/list of 2 integers.")

        Cout = b_tensor.shape[0]

        Hin = a_tensor.shape[2]
        Win = a_tensor.shape[3]

        Kh = kernel_size[0]
        Kw = kernel_size[1]

        Hout = math.floor((Hin - Kh + 2 * padding) / stride[0] + 1)
        Wout = math.floor((Win - Kw + 2 * padding) / stride[1] + 1)

        return (a_tensor.shape[0], Cout, Hout, Wout)

    @staticmethod
    def forward(
        out: "Tensor", a_tensor: "Tensor", b_tensor: "Tensor",
        kernel_size: tuple[int, ...], stride: tuple[int, int], padding: int
        ):
        a_realized = a_tensor.realize()
        b_realized = b_tensor.realize()
        c_conv(
            a_realized.c_tensor_ptr,
            b_realized.c_tensor_ptr,
            out.c_tensor_ptr,
            (ctypes.c_int * 2)(*kernel_size), # Convert tuple to C array pointer
            (ctypes.c_int * 2)(*stride),     # Convert tuple to C array pointer
            padding
        )

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: ctypes.POINTER(Conv2DBackwardExtras)):
        c_conv_grad_op(out_ptr, prev_ptrs, n_prev, extras)


class Dot(BOp):
    @staticmethod
    def calc_out_shape(*args: Any, **kwargs: Any) -> tuple[int, ...]:
        from fajr.core.tensor import Tensor
        if len(args) < 2:
            raise ValueError("Dot.calc_out_shape requires two Tensor operands.")
        a_tensor: Tensor = args[0]
        b_tensor: Tensor = args[1]

        if not isinstance(a_tensor, Tensor) or not isinstance(b_tensor, Tensor):
            raise TypeError("Dot operands must be Tensors.")

        if a_tensor.ndim == 1 and b_tensor.ndim == 1:
            if a_tensor.shape[0] != b_tensor.shape[0]:
                raise ValueError(f"Dot product of 1D tensors requires matching dimensions: {a_tensor.shape[0]} vs {b_tensor.shape[0]}")
            return ()

        if a_tensor.shape[-1] != b_tensor.shape[-1]:
            raise ValueError(f"Last dimensions must match for dot product contraction: {a_tensor.shape[-1]} vs {b_tensor.shape[-1]}")

        a_batch_shape = a_tensor.shape[:-1]
        b_batch_shape = b_tensor.shape[:-1]

        max_ndim_batch = max(len(a_batch_shape), len(b_batch_shape))
        padded_a_batch_shape = (1,) * (max_ndim_batch - len(a_batch_shape)) + a_batch_shape
        padded_b_batch_shape = (1,) * (max_ndim_batch - len(b_batch_shape)) + b_batch_shape

        result_batch_shape = []
        for dim1, dim2 in zip(padded_a_batch_shape, padded_b_batch_shape):
            if dim1 == dim2:
                result_batch_shape.append(dim1)
            elif dim1 == 1:
                result_batch_shape.append(dim2)
            elif dim2 == 1:
                result_batch_shape.append(dim1)
            else:
                raise ValueError(f"Batch shapes are not broadcastable for dot product: {a_tensor.shape} and {b_tensor.shape}")

        return tuple(result_batch_shape)

    @staticmethod
    def create_ctx_struct(*args: Any, **kwargs: Any) -> Tuple[Dict[str, Any], Any]:
        return {}, None

    @staticmethod
    def forward(out: "Tensor", a_tensor: "Tensor", b_tensor: "Tensor"):
        batch_shape = out.shape
        K = a_tensor.shape[-1]
        a_target_shape = batch_shape + (K,)
        b_target_shape = batch_shape + (K,)
        a_broadcasted = a_tensor.broadcast(a_target_shape).realize()
        b_broadcasted = b_tensor.broadcast(b_target_shape).realize()
        c_dot(a_broadcasted.c_tensor_ptr, b_broadcasted.c_tensor_ptr, out.c_tensor_ptr)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        c_dot_grad_op(out_ptr, prev_ptrs, n_prev, extras)

