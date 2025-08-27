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
        output_tensor = ctx.forward(*args, **kwargs)

        if output_tensor.requires_grad:
            from py.core import Node
            from py.core import Tensor

            # Add validation here
            if not output_tensor.validate():
                raise RuntimeError(
                    "Output tensor is in an invalid state before Node creation."
                )

            input_tensors = [arg for arg in args if isinstance(arg, (Tensor, CTensor))]

            # Explicitly keep references
            ctx._output_tensor_ref = output_tensor
            ctx._input_tensors_ref = input_tensors

            backward_fn = ctx._get_backward_fn_type()

            output_tensor._node = Node(
                out_tensor=output_tensor,
                input_tensors=input_tensors,
                backward_fn=backward_fn,
                extras=ctx.extras,
            )
        return output_tensor


# --- Binary Operations ---


class Add(Function):
    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return BackwardFnType(tensor_lib.add_grad_op)

    def forward(self, a: "Tensor", b: "Tensor") -> "Tensor":
        from py.core import Tensor

        out_tensor = Tensor(
            shape=a.shape, requires_grad=a.requires_grad or b.requires_grad
        )

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

    def forward(self, a: "Tensor", b: "Tensor") -> "Tensor":
        from ..core.tensor import Tensor

        out_tensor = Tensor(
            shape=a.shape, requires_grad=a.requires_grad or b.requires_grad
        )
        if isinstance(b, Tensor):
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

    def forward(self, a: "Tensor", b: float) -> "Tensor":
        from ..core.tensor import Tensor

        out_tensor = Tensor(shape=a.shape, requires_grad=a.requires_grad)
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

    def forward(self, a: "Tensor", b: "Tensor") -> "Tensor":
        from ..core.tensor import Tensor

        out_tensor = Tensor(
            shape=a.shape, requires_grad=a.requires_grad or b.requires_grad
        )
        if isinstance(b, Tensor):
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

    def forward(self, a: "Tensor", b: "Tensor") -> "Tensor":
        from ..core.tensor import Tensor

        out_tensor = Tensor(
            shape=a.shape, requires_grad=a.requires_grad or b.requires_grad
        )
        if isinstance(b, Tensor):
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

    def forward(self, a: "Tensor", b: float) -> "Tensor":  # b is scalar, a is tensor
        from ..core.tensor import Tensor

        out_tensor = Tensor(shape=a.shape, requires_grad=a.requires_grad)
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
    def forward(self, a: "Tensor", b: "Tensor") -> "Tensor":
        from ..core.tensor import Tensor

        N = a.shape[-2]
        M = a.shape[-1]

        if a.shape[-1] != b.shape[-2]:
            raise RuntimeError(
                f"Can't multiply tensors with shapes {a.shape} and {b.shape}"
            )

        K = a.shape[-1]
        P = b.shape[-1]

        out_c_tensor_ptr = c_malloc_tensor_empty()
        if not out_c_tensor_ptr:
            raise RuntimeError(
                "Failed to allocate empty C tensor for matmul operation."
            )

        c_matmul(a._c_tensor, b._c_tensor, out_c_tensor_ptr, N, K, P)

        out_tensor = Tensor(
            _c_tensor_ptr=out_c_tensor_ptr,
            requires_grad=a.requires_grad or b.requires_grad,
        )

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

    def forward(self, a: "Tensor") -> "Tensor":
        from ..core.tensor import Tensor

        out_tensor = Tensor(shape=a.shape, requires_grad=a.requires_grad)
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

    def forward(self, a: "Tensor") -> "Tensor":
        from ..core.tensor import Tensor

        out_tensor = Tensor(shape=a.shape, requires_grad=a.requires_grad)
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

    def forward(self, a: "Tensor") -> "Tensor":
        from ..core.tensor import Tensor

        out_tensor = Tensor(shape=a.shape, requires_grad=a.requires_grad)
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
    def forward(self, a: "Tensor") -> "Tensor":
        from ..core.tensor import Tensor

        out_tensor = Tensor(shape=a.shape, requires_grad=a.requires_grad)
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

    def forward(self, a: "Tensor") -> "Tensor":
        from ..core.tensor import Tensor

        out_tensor = Tensor(shape=a.shape, requires_grad=a.requires_grad)
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

    def forward(self, a: "Tensor") -> "Tensor":
        from ..core.tensor import Tensor

        out_tensor = Tensor(shape=a.shape, requires_grad=a.requires_grad)
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
    def forward(self, a: "Tensor", axis: int = 0, keepdim: bool = True) -> "Tensor":
        from ..core.tensor import Tensor

        out_c_tensor_ptr = c_malloc_tensor_empty()
        if not out_c_tensor_ptr:
            raise RuntimeError("Failed to allocate empty C tensor for sum operation.")
        c_sum(a._c_tensor, out_c_tensor_ptr, axis, keepdim)
        out_tensor = Tensor(
            _c_tensor_ptr=out_c_tensor_ptr, requires_grad=a.requires_grad
        )

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
    def forward(self, a: "Tensor", axis: int = 0, keepdim: bool = True) -> "Tensor":
        from ..core.tensor import Tensor

        out_c_tensor_ptr = c_malloc_tensor_empty()
        if not out_c_tensor_ptr:
            raise RuntimeError("Failed to allocate empty C tensor for mean operation.")
        c_mean(a._c_tensor, out_c_tensor_ptr, axis, keepdim)
        out_tensor = Tensor(
            _c_tensor_ptr=out_c_tensor_ptr, requires_grad=a.requires_grad
        )

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
    def forward(self, a: "Tensor", axis: int = 0, keepdim: bool = True) -> "Tensor":
        from ..core.tensor import Tensor

        out_c_tensor_ptr = c_malloc_tensor_empty()
        if not out_c_tensor_ptr:
            raise RuntimeError("Failed to allocate empty C tensor for max operation.")
        c_max(a._c_tensor, out_c_tensor_ptr, axis, keepdim)
        out_tensor = Tensor(
            _c_tensor_ptr=out_c_tensor_ptr, requires_grad=a.requires_grad
        )

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
    def forward(self, a: "Tensor", shape: list) -> "Tensor":
        from ..core.tensor import Tensor

        if a.numel() != Tensor.safe_c_numel(
            shape, len(shape)
        ):  # Use Tensor's numel for comparison
            raise RuntimeError(
                f"Unable to operate view as {a.numel()} != {Tensor.safe_c_numel(shape, len(shape))}"
            )
        out_c_tensor_ptr = c_malloc_tensor_empty()
        if not out_c_tensor_ptr:
            raise RuntimeError("Failed to allocate empty C tensor for view operation.")
        c_view(a._c_tensor, out_c_tensor_ptr, shape, len(shape))
        out_tensor = Tensor(
            _c_tensor_ptr=out_c_tensor_ptr, requires_grad=a.requires_grad
        )

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
    def forward(self, a: "Tensor", dim: int = 0) -> "Tensor":
        from ..core.tensor import Tensor

        if dim < 0:
            dim = a.ndim + dim + 1
        if dim > a.ndim:
            raise ValueError(f"Can't unsqueeze dim {dim}.")
        out_c_tensor_ptr = c_malloc_tensor_empty()
        if not out_c_tensor_ptr:
            raise RuntimeError(
                "Failed to allocate empty C tensor for unsqueeze operation."
            )
        c_unsqueeze(a._c_tensor, out_c_tensor_ptr, dim)
        out_tensor = Tensor(
            _c_tensor_ptr=out_c_tensor_ptr, requires_grad=a.requires_grad
        )

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
    def forward(self, a: "Tensor", dim: int = 0) -> "Tensor":
        from ..core.tensor import Tensor

        if dim < 0:
            dim = a.ndim + dim
        if dim >= a.ndim:
            raise ValueError(f"Can't squeeze dim {dim} as it doesn't exists.")
        if a.shape[dim] != 1:
            raise RuntimeError(
                f"Tensors can be squeezed only on shape[dim] = 1, dim = {a.shape[dim]}"
            )
        out_c_tensor_ptr = c_malloc_tensor_empty()
        if not out_c_tensor_ptr:
            raise RuntimeError(
                "Failed to allocate empty C tensor for squeeze operation."
            )
        c_squeeze(a._c_tensor, out_c_tensor_ptr, dim)
        out_tensor = Tensor(
            _c_tensor_ptr=out_c_tensor_ptr, requires_grad=a.requires_grad
        )

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
    def forward(self, a: "Tensor", n: int = -2, m: int = -1) -> "Tensor":
        from ..core.tensor import Tensor

        if n < 0:
            n = a.ndim + n
        if m < 0:
            m = a.ndim + m
        if n >= a.ndim or m >= a.ndim:
            raise ValueError(f"Can't transpose around non-exsiting axes {n},{m}")
        out_c_tensor_ptr = c_malloc_tensor_empty()
        if not out_c_tensor_ptr:
            raise RuntimeError(
                "Failed to allocate empty C tensor for transpose operation."
            )
        c_transpose(a._c_tensor, out_c_tensor_ptr, n, m)
        out_tensor = Tensor(
            _c_tensor_ptr=out_c_tensor_ptr, requires_grad=a.requires_grad
        )

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
    def forward(self, a: "Tensor", shape: list[int]) -> "Tensor":
        from ..core.tensor import Tensor

        if a.ndim != len(shape):
            raise ValueError(f"expand() error: Dimensionality mismatch")
        for i in range(a.ndim):
            if a.shape[i] != shape[i] and a.shape[i] != 1:
                raise RuntimeError(
                    f"expand() error: Can't expand dim {i} from {a.shape[i]} to {shape[i]}, Only dims of 1 can be expanded."
                )
        out_c_tensor_ptr = c_malloc_tensor_empty()
        if not out_c_tensor_ptr:
            raise RuntimeError(
                "Failed to allocate empty C tensor for expand operation."
            )
        c_expand(a._c_tensor, out_c_tensor_ptr, shape)
        out_tensor = Tensor(
            _c_tensor_ptr=out_c_tensor_ptr, requires_grad=a.requires_grad
        )

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        pass  # TODO: Implement expand_grad_op
