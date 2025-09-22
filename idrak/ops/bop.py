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
    def create_ctx_struct(*args, **kwargs) -> Tuple[Dict[str, Any], Any]:
        from idrak.core.tensor import Tensor
        a_operand: Optional[Tensor] = None
        if args and isinstance(args[0], Tensor):
            a_operand = args[0]
        else:
            raise TypeError("First operand for BOp must be a Tensor.")

        b_operand: Any = None
        if len(args) > 1:
            b_operand = args[1]
        elif 'scalar_val' in kwargs:
            b_operand = kwargs['scalar_val']
        
        forward_kwargs: Dict[str, Any] = {}
        backward_ctx: Any = None

        if isinstance(b_operand, (float, int)):
            forward_kwargs["scalar_val"] = float(b_operand)
            backward_ctx = ctypes.c_float(float(b_operand))

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
    def calc_out_shape(*args, **kwargs) -> tuple[int, ...]:
        from idrak.core.tensor import Tensor
        if not args:
            raise ValueError("calc_out_shape requires at least 'a' operand.")
        
        a_tensor: Tensor = args[0]
        b_operand: Any = None
        if len(args) > 1:
            b_operand = args[1]
        elif 'scalar_val' in kwargs:
            b_operand = kwargs['scalar_val']
        
        if not isinstance(a_tensor, Tensor):
            raise TypeError(f"First operand must be a Tensor, got {type(a_tensor)}")

        if isinstance(b_operand, (Tensor, CTensor)): # CTensor if passed directly as C_Tensor (unwrapped)
            return BOp.compute_broadcasted_shape(a_tensor.shape, Tensor._wrap_c_tensor_ptr(b_operand).shape if isinstance(b_operand, CTensor) else b_operand.shape)
        elif isinstance(b_operand, (float, int)):
            return a_tensor.shape
        else:
            raise ValueError(f"Second operand must be a Tensor or a scalar, got {type(b_operand)} (or None).")

class Add(BOp):
    @staticmethod
    def calc_out_shape(*args, **kwargs) -> tuple[int, ...]:
        return args[0].shape
    @staticmethod
    def forward(out: "Tensor", a_tensor: "Tensor", b_tensor: Optional["Tensor"] = None, scalar_val: Optional[float] = None):
        if b_tensor is not None:
            a_broadcasted = a_tensor # .broadcast(out.shape)
            b_broadcasted = b_tensor #.broadcast(out.shape)
            c_add(a_broadcasted.c_tensor_ptr, b_broadcasted.c_tensor_ptr, out.c_tensor_ptr)
        elif scalar_val is not None:
            scalar = ctypes.c_float(scalar_val)
            c_add_scalar(a_tensor.c_tensor_ptr, scalar, out.c_tensor_ptr)
        else:
            raise ValueError("Add operation requires either a Tensor or a scalar for its second operand.")

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        c_add_grad_op(out_ptr, prev_ptrs, n_prev, extras)

class Sub(BOp):
    @staticmethod
    def forward(out: "Tensor", a_tensor: "Tensor", b_tensor: Optional["Tensor"] = None, scalar_val: Optional[float] = None):
        if b_tensor is not None:
            a_broadcasted = a_tensor.broadcast(out.shape)
            b_broadcasted = b_tensor.broadcast(out.shape)
            c_sub(a_broadcasted.c_tensor_ptr, b_broadcasted.c_tensor_ptr, out.c_tensor_ptr)
        elif scalar_val is not None:
            scalar = ctypes.c_float(scalar_val)
            c_sub_scalar(a_tensor.c_tensor_ptr, scalar, out.c_tensor_ptr)
        else:
            raise ValueError("Sub operation requires either a Tensor or a scalar for its second operand.")

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        c_sub_grad_op(out_ptr, prev_ptrs, n_prev, extras)

class RSub(BOp):
    @staticmethod
    def create_ctx_struct(*args, **kwargs) -> Tuple[Dict[str, Any], Any]: # <- OVERRIDE
        a_operand: Any = None
        if args and not isinstance(args[0], Tensor):
            a_operand = args[0]
        else:
            raise TypeError("RSub (scalar - Tensor) operation expected first operand to be a scalar.")
        
        forward_kwargs: Dict[str, Any] = {"r_scalar_val": float(a_operand)}
        backward_ctx: Any = ctypes.c_float(float(a_operand))
        
        return forward_kwargs, backward_ctx

    @staticmethod
    def calc_out_shape(*args, **kwargs) -> tuple[int, ...]:
        if len(args) < 2:
            raise ValueError("RSub.calc_out_shape requires a scalar (a) and a Tensor (b).")
        
        a_scalar = args[0]
        b_tensor = args[1]
        
        if not isinstance(a_scalar, (float, int)):
            raise TypeError(f"First operand for RSub must be a scalar, got {type(a_scalar)}")
        if not isinstance(b_tensor, Tensor):
            raise TypeError(f"Second operand for RSub must be a Tensor, got {type(b_tensor)}")
        
        return b_tensor.shape

    @staticmethod
    def forward(out: "Tensor", b_tensor: "Tensor", r_scalar_val: float = None):
        if r_scalar_val is not None:
            scalar = ctypes.c_float(r_scalar_val)
            c_rsub_scalar(scalar, b_tensor.c_tensor_ptr, out.c_tensor_ptr)
        else:
            raise ValueError("RSub operation requires 'r_scalar_val' in forward kwargs.")

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        c_rsub_grad_op(out_ptr, prev_ptrs, n_prev, extras)

class Mul(BOp):
    @staticmethod
    def forward(out: "Tensor", a_tensor: "Tensor", b_tensor: Optional["Tensor"] = None, scalar_val: Optional[float] = None):
        if b_tensor is not None:
            a_broadcasted = a_tensor.broadcast(out.shape)
            b_broadcasted = b_tensor.broadcast(out.shape)
            c_mul(a_broadcasted.c_tensor_ptr, b_broadcasted.c_tensor_ptr, out.c_tensor_ptr)
        elif scalar_val is not None:
            scalar = ctypes.c_float(scalar_val)
            c_mul_scalar(a_tensor.c_tensor_ptr, scalar, out.c_tensor_ptr)
        else:
            raise ValueError("Mul operation requires either a Tensor or a scalar for its second operand.")

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        c_mul_grad_op(out_ptr, prev_ptrs, n_prev, extras)

class Div(BOp):
    @staticmethod
    def forward(out: "Tensor", a_tensor: "Tensor", b_tensor: Optional["Tensor"] = None, scalar_val: Optional[float] = None):
        if b_tensor is not None:
            a_broadcasted = a_tensor.broadcast(out.shape)
            b_broadcasted = b_tensor.broadcast(out.shape)
            c_div(a_broadcasted.c_tensor_ptr, b_broadcasted.c_tensor_ptr, out.c_tensor_ptr)
        elif scalar_val is not None:
            scalar = ctypes.c_float(scalar_val)
            c_div_scalar(a_tensor.c_tensor_ptr, scalar, out.c_tensor_ptr)
        else:
            raise ValueError("Div operation requires either a Tensor or a scalar for its second operand.")

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        c_div_grad_op(out_ptr, prev_ptrs, n_prev, extras)

class RDiv(BOp):
    @staticmethod
    def create_ctx_struct(*args, **kwargs) -> Tuple[Dict[str, Any], Any]:
        from idrak.core.tensor import Tensor
        a_operand: Any = None
        if args and not isinstance(args[0], Tensor):
            a_operand = args[0]
        else:
            raise TypeError("RDiv (scalar / Tensor) operation expected first operand to be a scalar.")
        
        forward_kwargs: Dict[str, Any] = {"r_scalar_val": float(a_operand)}
        backward_ctx: Any = ctypes.c_float(float(a_operand))
        
        return forward_kwargs, backward_ctx

    @staticmethod
    def calc_out_shape(*args, **kwargs) -> tuple[int, ...]:
        if len(args) < 2:
            raise ValueError("RDiv.calc_out_shape requires a scalar (a) and a Tensor (b).")
        
        a_scalar = args[0]
        b_tensor = args[1]
        
        if not isinstance(a_scalar, (float, int)):
            raise TypeError(f"First operand for RDiv must be a scalar, got {type(a_scalar)}")
        if not isinstance(b_tensor, Tensor):
            raise TypeError(f"Second operand for RDiv must be a Tensor, got {type(b_tensor)}")
        
        return b_tensor.shape

    @staticmethod
    def forward(out: "Tensor", b_tensor: "Tensor", r_scalar_val: Optional[float] = None):
        if r_scalar_val is not None:
            scalar = ctypes.c_float(r_scalar_val)
            c_rdiv_scalar(scalar, b_tensor.c_tensor_ptr, out.c_tensor_ptr)
        else:
            raise ValueError("RDiv operation requires 'r_scalar_val' in forward kwargs.")

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        c_rdiv_grad_op(out_ptr, prev_ptrs, n_prev, extras)

class Pow(BOp):
    @staticmethod
    def forward(out: "Tensor", a_tensor: "Tensor", b_tensor: Optional["Tensor"] = None, scalar_val: Optional[float] = None):
        if b_tensor is not None:
            a_broadcasted = a_tensor.broadcast(out.shape)
            b_broadcasted = b_tensor.broadcast(out.shape)
            c_pow(a_broadcasted.c_tensor_ptr, b_broadcasted.c_tensor_ptr, out.c_tensor_ptr)
        elif scalar_val is not None:
            scalar = ctypes.c_float(scalar_val)
            c_pow_scalar(a_tensor.c_tensor_ptr, scalar, out.c_tensor_ptr)
        else:
            raise ValueError("Pow operation requires either a Tensor or a scalar for its second operand.")
    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        c_pow_grad_op(out_ptr, prev_ptrs, n_prev, extras)

class MatMul(BOp):
    @staticmethod
    def calc_out_shape(*args, **kwargs) -> tuple[int, ...]:
        from idrak.core.tensor import Tensor
        if len(args) < 2:
            raise ValueError("MatMul.calc_out_shape requires two Tensor operands.")
        a_tensor = args[0]
        b_tensor = args[1]

        if not isinstance(a_tensor, Tensor) or not isinstance(b_tensor, Tensor):
            raise TypeError("MatMul operands must be Tensors.")

        a_effective_shape = a_tensor.shape[:-2] if a_tensor.ndim >= 2 else ()
        b_effective_shape = b_tensor.shape[:-2] if b_tensor.ndim >= 2 else ()

        a_K = a_tensor.shape[-1]
        b_K = b_tensor.shape[-2] if b_tensor.ndim >= 2 else b_tensor.shape[-1]

        if a_K != b_K:
            raise ValueError(f"Matrix multiplication dimensions are incompatible: {a_tensor.shape} and {b_tensor.shape}")

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
                raise ValueError(f"Batch shapes are not broadcastable: {a_tensor.shape} and {b_tensor.shape}")

        a_N = a_tensor.shape[-2] if a_tensor.ndim >= 2 else 1
        b_M = b_tensor.shape[-1] if b_tensor.ndim >= 2 else 1

        if a_tensor.ndim == 1 and b_tensor.ndim == 1:
            return (1,)
        elif a_tensor.ndim == 1:
            return tuple(result_batch_shape) + (b_M,)
        elif b_tensor.ndim == 1:
            return tuple(result_batch_shape) + (a_N,)
        else:
            return tuple(result_batch_shape) + (a_N, b_M)

    @staticmethod
    def create_ctx_struct(*args, **kwargs) -> Tuple[Dict[str, Any], Any]:
        return {}, None

    @staticmethod
    def forward(out: "Tensor", a_tensor: "Tensor", b_tensor: "Tensor"):
        out_batch_shape = out.shape[:-2] if out.ndim >= 2 else ()

        a_target_shape = out_batch_shape
        if a_tensor.ndim == 1:
            a_target_shape += (1, a_tensor.shape[0])
        else:
            a_target_shape += a_tensor.shape[-2:]

        b_target_shape = out_batch_shape
        if b_tensor.ndim == 1:
            b_target_shape += (b_tensor.shape[0], 1)
        else:
            b_target_shape += b_tensor.shape[-2:]

        # Perform broadcasting
        a_broadcasted = a_tensor.broadcast(a_target_shape)
        b_broadcasted = b_tensor.broadcast(b_target_shape)

        N = a_broadcasted.shape[-2]
        K = a_broadcasted.shape[-1]
        M = b_broadcasted.shape[-1]

        if a_broadcasted.shape[-1] != b_broadcasted.shape[-2]:
            raise RuntimeError(f"Matrix multiplication dimensions are incompatible after broadcasting: {a_broadcasted.shape} and {b_broadcasted.shape}")

        c_matmul(a_broadcasted.c_tensor_ptr, b_broadcasted.c_tensor_ptr, out.c_tensor_ptr, N=N, K=K, P=M)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        c_matmul_grad_op(out_ptr, prev_ptrs, n_prev, extras)


class Conv2D(BOp):
    @staticmethod
    def create_ctx_struct(*args, **kwargs) -> Tuple[Dict[str, Any], Any]:
        from idrak.core.tensor import Tensor
        if len(args) < 2:
            raise ValueError("Conv2D.create_ctx_struct requires at least two Tensor operands (input, kernel).")
        
        a_tensor: Tensor = args[0]
        b_tensor: Tensor = args[1]

        kernel_size = kwargs.get("kernel_size")
        stride = kwargs.get("stride")
        padding = kwargs.get("padding")

        if kernel_size is None or stride is None or padding is None:
            raise ValueError("Conv2D.create_ctx_struct requires kernel_size, stride, and padding as keyword arguments.")

        Hin = a_tensor.shape[2]
        Win = a_tensor.shape[3]

        Kh = kernel_size[0]
        Kw = kernel_size[1]
        
        Sh = stride[0]
        Sw = stride[1]

        Hout = math.floor((Hin - Kh + 2 * padding + 1) / Sh)
        Wout = math.floor((Win - Kw + 2 * padding + 1) / Sw)

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
    def calc_out_shape(*args, **kwargs) -> tuple[int, ...]:
        if len(args) < 2:
            raise ValueError("Conv2D.calc_out_shape requires at least two Tensor operands (input, kernel).")
        a_tensor: Tensor = args[0]
        b_tensor: Tensor = args[1]

        kernel_size = kwargs.get("kernel_size")
        stride = kwargs.get("stride")
        padding = kwargs.get("padding")

        if kernel_size is None or stride is None or padding is None:
            raise ValueError("Conv2D.calc_out_shape requires kernel_size, stride, and padding as keyword arguments.")

        Cout = b_tensor.shape[0]

        Hin = a_tensor.shape[2]
        Win = a_tensor.shape[3]

        Kh = kernel_size[0]
        Kw = kernel_size[1]

        Hout = math.floor((Hin - Kh + 2 * padding + 1) / stride[0])
        Wout = math.floor((Win - Kw + 2 * padding + 1) / stride[1])

        return (a_tensor.shape[0], Cout, Hout, Wout)

    @staticmethod
    def forward(
        out: "Tensor", a_tensor: "Tensor", b_tensor: "Tensor",
        kernel_size: tuple[int, ...], stride: tuple[int, int], padding: int
        ):
        c_conv(a_tensor.c_tensor_ptr, b_tensor.c_tensor_ptr, out.c_tensor_ptr, kernel_size, stride, padding)
    
    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: ctypes.POINTER(Conv2DBackwardExtras)):
        c_conv_grad_op(out_ptr, prev_ptrs, n_prev, extras)

class Dot(BOp):
    @staticmethod
    def calc_out_shape(*args, **kwargs) -> tuple[int, ...]:
        from idrak.core.tensor import Tensor
        if len(args) < 2:
            raise ValueError("Dot.calc_out_shape requires two Tensor operands.")
        a_tensor = args[0]
        b_tensor = args[1]

        if not isinstance(a_tensor, Tensor) or not isinstance(b_tensor, Tensor):
            raise TypeError("Dot operands must be Tensors.")

        if a_tensor.ndim == 1 and b_tensor.ndim == 1:
            if a_tensor.shape[0] != b_tensor.shape[0]:
                raise ValueError(f"Dot product of 1D tensors requires matching dimensions: {a_tensor.shape[0]} vs {b_tensor.shape[0]}")
            return (1,) # Scalar output

        a_batch_shape = a_tensor.shape[:-1]
        b_batch_shape = b_tensor.shape[:-1]
        
        if a_tensor.shape[-1] != b_tensor.shape[-1]:
            raise ValueError(f"Last dimensions must match for dot product contraction: {a_tensor.shape[-1]} vs {b_tensor.shape[-1]}")

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
        
        return tuple(result_batch_shape) if result_batch_shape else (1,)

    @staticmethod
    def create_ctx_struct(*args, **kwargs) -> Tuple[Dict[str, Any], Any]:
        return {}, None

    @staticmethod
    def forward(out: "Tensor", a_tensor: "Tensor", b_tensor: "Tensor"):
        c_dot(a_tensor.c_tensor_ptr, b_tensor.c_tensor_ptr, out.c_tensor_ptr)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        c_dot_grad_op(out_ptr, prev_ptrs, n_prev, extras)

