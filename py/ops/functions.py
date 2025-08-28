from __future__ import annotations
import ctypes
from typing import Any, Tuple, Optional, List
import sys

from ..elnawah_bindings.c_wrapper_functions import (
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
    c_relu,
    c_log,
    c_exp,
    c_softmax,
    c_abs,
    c_neg,
    c_sum,
    c_mean,
    c_max,
    c_view,
    c_unsqueeze,
    c_squeeze,
    c_transpose,
    c_expand,
    c_add_grad_op,
    c_sub_grad_op,
    c_rsub_grad_op,
    c_mul_grad_op,
    c_div_grad_op,
    c_rdiv_grad_op,
    c_relu_grad_op,
    c_log_grad_op,
    c_exp_grad_op,
    c_abs_grad_op,
    c_neg_grad_op,
    c_malloc_tensor_empty,
)
from ..elnawah_bindings.ctypes_definitions import CTensor, BackwardFnType
from ..elnawah_bindings.c_library_loader import tensor_lib
from .lazy_ops import (
    LazyAdd,
    LazySub,
    LazyRSub,
    LazyMul,
    LazyDiv,
    LazyRDiv,
    LazyMatMul,
    LazyReLU,
    LazyLog,
    LazyExp,
    LazySoftmax,
    LazyAbs,
    LazyNeg,
    LazySum,
    LazyMean,
    LazyMax,
    LazyView,
    LazyUnsqueeze,
    LazySqueeze,
    LazyTranspose,
    LazyExpand,
)


class Function:
    def __init__(self):
        self.saved_tensors: Tuple["Tensor", ...] = ()
        self.extras: Any = None
        self._output_tensor_ref: Optional["Tensor"] = None
        self._input_tensors_ref: List["Tensor"] = []

    def forward(self, *args: Any, **kwargs: Any) -> "Tensor":
        raise NotImplementedError

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        raise NotImplementedError

    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        """
        Returns the appropriate BackwardFnType for this operation.
        Subclasses should override this if they use a C-level backward function
        instead of their own Python backward method.
        """
        return BackwardFnType(self.backward)

    def _calc_output_shape(self, *args: Any, **kwargs: Any) -> Tuple[int, ...]:
        """
        Calculates the output shape of the operation without performing the computation.
        Subclasses must override this method.
        """
        raise NotImplementedError

    @classmethod
    def apply(cls, *args: Any, **kwargs: Any) -> "Tensor":
        ctx = cls()
        from py.core.tensor import Tensor

        requires_grad = any(
            isinstance(arg, (Tensor, CTensor)) and arg.requires_grad for arg in args
        )

        # Calculate output shape using the new method
        output_shape = ctx.lazy_op_class().calculate_output_shape(*args, **kwargs)

        output_tensor = Tensor(shape=output_shape, requires_grad=requires_grad)

        if output_tensor.requires_grad:
            from py.core.node import Node

            input_tensors = [arg for arg in args if isinstance(arg, (Tensor, CTensor))]

            ctx._output_tensor_ref = output_tensor
            ctx._input_tensors_ref = input_tensors

            backward_fn = ctx._get_backward_fn_type()

            output_tensor._node = Node(
                out_tensor=output_tensor,
                input_tensors=input_tensors,
                forward_fn=ctx._execute_forward,
                forward_args=(output_tensor, *input_tensors),
                forward_kwargs=kwargs,
                backward_fn=backward_fn,
                extras=ctx.extras,
            )
        else:
            # For non-grad tensors, execute immediately
            result = ctx._execute_forward(output_tensor, *args, **kwargs)
            output_tensor = result

        return output_tensor


# --- Binary Operations ---


class Add(Function):
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
    lazy_op_class = LazySub

    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.sub_grad_op)

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", b: "Tensor"
    ) -> "Tensor":
        from ..core.tensor import Tensor

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
    lazy_op_class = LazyRSub

    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.rsub_grad_op)

    def _execute_forward(self, out_tensor: "Tensor", a: "Tensor", b: float) -> "Tensor":
        from ..core.tensor import Tensor

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
    lazy_op_class = LazyMul

    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.mul_grad_op)

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", b: "Tensor"
    ) -> "Tensor":
        from ..core.tensor import Tensor

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
    lazy_op_class = LazyDiv

    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.div_grad_op)

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", b: "Tensor"
    ) -> "Tensor":
        from ..core.tensor import Tensor

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
    lazy_op_class = LazyRDiv

    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.rdiv_grad_op)

    def _execute_forward(self, out_tensor: "Tensor", a: "Tensor", b: float) -> "Tensor":
        from ..core.tensor import Tensor

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
    lazy_op_class = LazyMatMul

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", b: "Tensor"
    ) -> "Tensor":
        from ..core.tensor import Tensor

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
        # Matmul backward is more complex, will need to implement c_matmul_grad_op
        # For now, just a placeholder
        pass  # TODO: Implement matmul_grad_op


# --- Unary Operations ---


class ReLU(Function):
    lazy_op_class = LazyReLU

    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.relu_grad_op)

    def _execute_forward(self, out_tensor: "Tensor", a: "Tensor") -> "Tensor":
        from ..core.tensor import Tensor

        c_relu(a._c_tensor, out_tensor._c_tensor)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        c_relu_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)


class Log(Function):
    lazy_op_class = LazyLog

    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.log_grad_op)

    def _execute_forward(self, out_tensor: "Tensor", a: "Tensor") -> "Tensor":
        from py.core.tensor import Tensor

        c_log(a._c_tensor, out_tensor._c_tensor)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        c_log_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)


class Exp(Function):
    lazy_op_class = LazyExp

    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.exp_grad_op)

    def _execute_forward(self, out_tensor: "Tensor", a: "Tensor") -> "Tensor":
        from ..core.tensor import Tensor

        c_exp(a._c_tensor, out_tensor._c_tensor)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        c_exp_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)


class Softmax(Function):
    lazy_op_class = LazySoftmax

    def _execute_forward(self, out_tensor: "Tensor", a: "Tensor") -> "Tensor":
        from ..core.tensor import Tensor

        c_softmax(a._c_tensor, out_tensor._c_tensor)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        # Softmax backward is more complex, will need to implement c_softmax_grad_op
        pass  # TODO: Implement softmax_grad_op


class Abs(Function):
    lazy_op_class = LazyAbs

    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.abs_grad_op)

    def _execute_forward(self, out_tensor: "Tensor", a: "Tensor") -> "Tensor":
        from ..core.tensor import Tensor

        c_abs(a._c_tensor, out_tensor._c_tensor)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        c_abs_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)


class Neg(Function):
    lazy_op_class = LazyNeg

    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.neg_grad_op)

    def _execute_forward(self, out_tensor: "Tensor", a: "Tensor") -> "Tensor":
        from ..core.tensor import Tensor

        c_neg(a._c_tensor, out_tensor._c_tensor)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        c_neg_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)


# --- Reduction Operations ---


class Sum(Function):
    lazy_op_class = LazySum

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", axis: int = 0, keepdim: bool = True
    ) -> "Tensor":
        from ..core.tensor import Tensor

        c_sum(a._c_tensor, out_tensor._c_tensor, axis, keepdim)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        pass  # TODO: Implement sum_grad_op


class Mean(Function):
    lazy_op_class = LazyMean

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", axis: int = 0, keepdim: bool = True
    ) -> "Tensor":
        from ..core.tensor import Tensor

        c_mean(a._c_tensor, out_tensor._c_tensor, axis, keepdim)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        pass  # TODO: Implement mean_grad_op


class Max(Function):
    lazy_op_class = LazyMax

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", axis: int = 0, keepdim: bool = True
    ) -> "Tensor":
        from ..core.tensor import Tensor

        c_max(a._c_tensor, out_tensor._c_tensor, axis, keepdim)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        pass  # TODO: Implement max_grad_op


# --- Movement Operations ---


class View(Function):
    lazy_op_class = LazyView

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", shape: list
    ) -> "Tensor":
        from ..core.tensor import Tensor

        c_view(a._c_tensor, out_tensor._c_tensor, shape, len(shape))

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        pass  # TODO: Implement view_grad_op


class Unsqueeze(Function):
    lazy_op_class = LazyUnsqueeze

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", dim: int = 0
    ) -> "Tensor":
        from ..core.tensor import Tensor

        c_unsqueeze(a._c_tensor, out_tensor._c_tensor, dim)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        pass  # TODO: Implement unsqueeze_grad_op


class Squeeze(Function):
    lazy_op_class = LazySqueeze

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", dim: int = 0
    ) -> "Tensor":
        from ..core.tensor import Tensor

        c_squeeze(a._c_tensor, out_tensor._c_tensor, dim)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        pass  # TODO: Implement squeeze_grad_op


class Transpose(Function):
    lazy_op_class = LazyTranspose

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", n: int = -2, m: int = -1
    ) -> "Tensor":
        from ..core.tensor import Tensor

        c_transpose(a._c_tensor, out_tensor._c_tensor, n, m)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        pass  # TODO: Implement transpose_grad_op


class Expand(Function):
    lazy_op_class = LazyExpand

    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return None

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", shape: list[int]
    ) -> "Tensor":
        from ..core.tensor import Tensor

        c_expand(a._c_tensor, out_tensor._c_tensor, shape)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        pass  # TODO: Implement expand_grad_op
