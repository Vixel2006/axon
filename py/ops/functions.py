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
    c_malloc_tensor_empty,  # Added missing import
)
from ..elnawah_bindings.ctypes_definitions import CTensor, BackwardFnType
from ..elnawah_bindings.c_library_loader import tensor_lib


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

    @classmethod
    def apply(cls, *args: Any, **kwargs: Any) -> "Tensor":
        ctx = cls()
        from py.core.tensor import Tensor

        requires_grad = any(
            isinstance(arg, (Tensor, CTensor)) and arg.requires_grad for arg in args
        )

        output_shape = None
        for arg in args:
            if isinstance(arg, (Tensor, CTensor)) and arg.shape is not None:
                output_shape = arg.shape
                break

        if output_shape is not None:
            output_tensor = Tensor(shape=output_shape, requires_grad=requires_grad)
        else:
            output_tensor = Tensor(requires_grad=requires_grad)

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
    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.add_grad_op)

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", b: "Tensor"
    ) -> "Tensor":
        from py.core.tensor import Tensor

        if not hasattr(out_tensor, "_shape") or out_tensor._shape is None:
            out_tensor._shape = a.shape

        if isinstance(b, (Tensor, CTensor)):
            if a.shape != b.shape:
                raise RuntimeError(
                    f"Can't add tensors with shapes {a.shape} and {b.shape}"
                )
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
    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.sub_grad_op)

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", b: "Tensor"
    ) -> "Tensor":
        from ..core.tensor import Tensor

        # Ensure out_tensor has the correct shape
        if not hasattr(out_tensor, "_shape") or out_tensor._shape is None:
            out_tensor._shape = a.shape

        if isinstance(b, (Tensor, CTensor)):
            if a.shape != b.shape:
                raise RuntimeError(
                    f"Can't subtract tensors with shapes {a.shape} and {b.shape}"
                )
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
    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.rsub_grad_op)

    def _execute_forward(self, out_tensor: "Tensor", a: "Tensor", b: float) -> "Tensor":
        from ..core.tensor import Tensor

        # Ensure out_tensor has the correct shape
        if not hasattr(out_tensor, "_shape") or out_tensor._shape is None:
            out_tensor._shape = a.shape

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
    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.mul_grad_op)

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", b: "Tensor"
    ) -> "Tensor":
        from ..core.tensor import Tensor

        # Ensure out_tensor has the correct shape
        if not hasattr(out_tensor, "_shape") or out_tensor._shape is None:
            out_tensor._shape = a.shape

        if isinstance(b, (Tensor, CTensor)):
            if a.shape != b.shape:
                raise RuntimeError(
                    f"Can't multiply tensors with shapes {a.shape} and {b.shape}"
                )
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
    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.div_grad_op)

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", b: "Tensor"
    ) -> "Tensor":
        from ..core.tensor import Tensor

        # Ensure out_tensor has the correct shape
        if not hasattr(out_tensor, "_shape") or out_tensor._shape is None:
            out_tensor._shape = a.shape

        if isinstance(b, (Tensor, CTensor)):
            if a.shape != b.shape:
                raise RuntimeError(
                    f"Can't divide tensors with shapes {a.shape} and {b.shape}"
                )
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
    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.rdiv_grad_op)

    def _execute_forward(self, out_tensor: "Tensor", a: "Tensor", b: float) -> "Tensor":
        from ..core.tensor import Tensor

        # Ensure out_tensor has the correct shape
        if not hasattr(out_tensor, "_shape") or out_tensor._shape is None:
            out_tensor._shape = a.shape

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

        # Calculate output shape
        output_shape = list(a.shape[:-1]) + [P]

        # Update output tensor shape
        out_tensor._shape = tuple(output_shape)

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
    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.relu_grad_op)

    def _execute_forward(self, out_tensor: "Tensor", a: "Tensor") -> "Tensor":
        from ..core.tensor import Tensor

        # Ensure out_tensor has the correct shape
        if not hasattr(out_tensor, "_shape") or out_tensor._shape is None:
            out_tensor._shape = a.shape

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
    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.log_grad_op)

    def _execute_forward(self, out_tensor: "Tensor", a: "Tensor") -> "Tensor":
        from py.core.tensor import Tensor

        # Ensure out_tensor has the correct shape
        if not hasattr(out_tensor, "_shape") or out_tensor._shape is None:
            out_tensor._shape = a.shape

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
    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.exp_grad_op)

    def _execute_forward(self, out_tensor: "Tensor", a: "Tensor") -> "Tensor":
        from ..core.tensor import Tensor

        # Ensure out_tensor has the correct shape
        if not hasattr(out_tensor, "_shape") or out_tensor._shape is None:
            out_tensor._shape = a.shape

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
    def _execute_forward(self, out_tensor: "Tensor", a: "Tensor") -> "Tensor":
        from ..core.tensor import Tensor

        # Ensure out_tensor has the correct shape
        if not hasattr(out_tensor, "_shape") or out_tensor._shape is None:
            out_tensor._shape = a.shape

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
    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.abs_grad_op)

    def _execute_forward(self, out_tensor: "Tensor", a: "Tensor") -> "Tensor":
        from ..core.tensor import Tensor

        # Ensure out_tensor has the correct shape
        if not hasattr(out_tensor, "_shape") or out_tensor._shape is None:
            out_tensor._shape = a.shape

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
    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.neg_grad_op)

    def _execute_forward(self, out_tensor: "Tensor", a: "Tensor") -> "Tensor":
        from ..core.tensor import Tensor

        # Ensure out_tensor has the correct shape
        if not hasattr(out_tensor, "_shape") or out_tensor._shape is None:
            out_tensor._shape = a.shape

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
    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", shape: list
    ) -> "Tensor":
        from ..core.tensor import Tensor

        if a.numel() != Tensor.safe_c_numel(shape, len(shape)):
            raise RuntimeError(
                f"Unable to operate view as {a.numel()} != {Tensor.safe_c_numel(shape, len(shape))}"
            )

        # Update output tensor shape
        out_tensor._shape = tuple(shape)

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
    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", dim: int = 0
    ) -> "Tensor":
        from ..core.tensor import Tensor

        if dim < 0:
            dim = a.ndim + dim + 1
        if dim > a.ndim:
            raise ValueError(f"Can't unsqueeze dim {dim}.")

        # Calculate new shape
        new_shape = list(a.shape)
        new_shape.insert(dim, 1)
        out_tensor._shape = tuple(new_shape)

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
    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", dim: int = 0
    ) -> "Tensor":
        from ..core.tensor import Tensor

        if dim < 0:
            dim = a.ndim + dim
        if dim >= a.ndim:
            raise ValueError(f"Can't squeeze dim {dim} as it doesn't exist.")
        if a.shape[dim] != 1:
            raise RuntimeError(
                f"Tensors can be squeezed only on shape[dim] = 1, dim = {a.shape[dim]}"
            )

        # Calculate new shape
        new_shape = list(a.shape)
        new_shape.pop(dim)
        out_tensor._shape = tuple(new_shape)

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
    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", n: int = -2, m: int = -1
    ) -> "Tensor":
        from ..core.tensor import Tensor

        if n < 0:
            n = a.ndim + n
        if m < 0:
            m = a.ndim + m
        if n >= a.ndim or m >= a.ndim:
            raise ValueError(f"Can't transpose around non-existing axes {n},{m}")

        # Calculate new shape after transpose
        new_shape = list(a.shape)
        new_shape[n], new_shape[m] = new_shape[m], new_shape[n]
        out_tensor._shape = tuple(new_shape)

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
    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return None

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", shape: list[int]
    ) -> "Tensor":
        from ..core.tensor import Tensor

        if a.ndim != len(shape):
            raise ValueError(f"expand() error: Dimensionality mismatch")
        for i in range(a.ndim):
            if a.shape[i] != shape[i] and a.shape[i] != 1:
                raise RuntimeError(
                    f"expand() error: Can't expand dim {i} from {a.shape[i]} to {shape[i]}, Only dims of 1 can be expanded."
                )

        # Update output tensor shape
        out_tensor._shape = tuple(shape)

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
