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
    c_sum_grad_op,
    c_mean_grad_op,
    c_max_grad_op,
    c_broadcast,
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
    LazyBroadcast
)


class Function:
    """
    Base class for differentiable operations in the autograd engine.

    Each subclass represents a single differentiable operation and is
    responsible for:
      - Defining the forward computation (`_execute_forward`)
      - Registering a backward function (`_get_backward_fn_type`)
      - Optionally saving tensors for use in backward

    Attributes:
        saved_tensors (Tuple[Tensor]): Tensors saved for backward.
        extras (Any): Extra data passed to forward/backward (e.g., scalars).
        _output_tensor_ref (Tensor): Reference to the output tensor.
        _input_tensors_ref (List[Tensor]): References to input tensors.
    """

    def __init__(self):
        self.saved_tensors: Tuple["Tensor", ...] = ()
        self.extras: Any = None
        self._output_tensor_ref: Optional["Tensor"] = None
        self._input_tensors_ref: List["Tensor"] = []

    def forward(self, *args: Any, **kwargs: Any) -> "Tensor":
        """
        Defines the forward computation.

        Args:
            *args: Input tensors or scalars.
            **kwargs: Additional parameters.

        Returns:
            Tensor: The result of the operation.
        """
        raise NotImplementedError

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        """
        Defines the backward computation.

        Args:
            *out_tensor_ptr: The tensor outputed from the operation
            **prev_tensor_ptr: Tensors we put into the operation.
            n_prev: number of tensors that get into the operation

        Returns:
            Tensor: The result of the operation.
        """
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
        """
        apply:
        Entry point for applying an operation within the autograd system.

        Description:
        - Creates a new output Tensor resulting from applying the operation.
        - If any input tensor requires gradients, a new Node is created to
          capture the computation graph for lazy execution and backpropagation.
        - If no gradients are required, the forward computation is executed
          immediately in eager mode.

        Parameters:
        - *args (Any): Input arguments to the operation (can include Tensors or raw values).
        - **kwargs (Any): Keyword arguments to the operation.

        Returns:
        - Tensor: The output tensor produced by the operation.

        Effects:
        - When gradients are required, attaches a Node to the output tensor to record
          dependencies and backward function.
        - When gradients are not required, executes forward immediately and stores
          the result directly in the output tensor.
        """
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
    """
    Elementwise addition.

    Forward:
        out = a + b

    Backward:
        dL/da = dL/dout
        dL/db = dL/dout

    Notes:
        - Supports tensor-tensor and tensor-scalar addition.
        - Registers C-level backward function `c_add_grad_op`.
    """

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
    """
    Elementwise substration.

    Forward:
        out = a - b

    Backward:
        dL/da = dL/dout
        dL/db = -dL/dout

    Notes:
        - Supports tensor-tensor and tensor-scalar addition.
        - Registers C-level backward function `c_sub_grad_op`.
    """

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
    """
    Reverse elementwise substraction (scalar - tensor).

    Forward:
        out = b - a

    Backward:
        dL/da = -dL/dout

    Notes:
        - Only supports scalar substracted by tensor.
        - Uses C-level kernel `c_rsub_scalar`.
        - Registers C-level backward function `c_rsub_grad_op`.
    """

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
    """
    Elementwise multiplication.

    Forward:
        out = a * b

    Backward:
        dL/da = b * dL/dout
        dL/db = a * dL/dout

    Notes:
        - Supports tensor-tensor and tensor-scalar multiplication.
        - Uses C-level kernels `c_mul` and `c_mul_scalar`.
        - Registers C-level backward function `c_mul_grad_op`.
    """

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
    """
    Elementwise division.

    Forward:
        out = a / b

    Backward:
        dL/da = (1 / b) * dL/dout
        dL/db = -(a / b^2) * dL/dout

    Notes:
        - Supports tensor-tensor and tensor-scalar division.
        - Uses C-level kernels `c_div` and `c_div_scalar`.
        - Registers C-level backward function `c_div_grad_op`.
    """

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
    """
    Reverse elementwise division (scalar / tensor).

    Forward:
        out = b / a

    Backward:
        dL/da = -(b / a^2) * dL/dout

    Notes:
        - Only supports scalar divided by tensor.
        - Uses C-level kernel `c_rdiv_scalar`.
        - Registers C-level backward function `c_rdiv_grad_op`.
    """

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
    """
    Matrix multiplication.

    Forward:
        out = a @ b

    Backward:
        dL/da = dL/dout @ b^T
        dL/db = a^T @ dL/dout

    Notes:
        - Requires that a.shape[-1] == b.shape[-2].
        - Uses C-level kernel `c_matmul`.
        - Backward not yet implemented (TODO: `c_matmul_grad_op`).
    """

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
    """
    Rectified Linear Unit (ReLU).

    Forward:
        out = max(0, a)

    Backward:
        dL/da = dL/dout if a > 0 else 0

    Notes:
        Uses efficient C SIMD kernels for forward and backward.
    """

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
    """
    Natural logarithm.

    Forward:
        out = log(a)

    Backward:
        dL/da = (1 / a) * dL/dout

    Notes:
        - Uses C-level SIMD kernel `c_log`.
        - Registers C-level backward function `c_log_grad_op`.
    """

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
    """
    Exponential function.

    Forward:
        out = exp(a)

    Backward:
        dL/da = exp(a) * dL/dout

    Notes:
        - Uses C-level SIMD kernel `c_exp`.
        - Registers C-level backward function `c_exp_grad_op`.
    """

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
    """
    Softmax activation (over the last dimension).

    Forward:
        out[i] = exp(a[i]) / Σ_j exp(a[j])

    Backward:
        dL/da = out * (dL/dout - Σ_j(dL/dout_j * out_j))

    Notes:
        - Uses C-level kernel `c_softmax`.
        - Backward not yet implemented (TODO: `c_softmax_grad_op`).
    """

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
    """
    Absolute value.

    Forward:
        out = |a|

    Backward:
        dL/da = sign(a) * dL/dout
        where sign(a) = +1 if a > 0, -1 if a < 0, else 0

    Notes:
        - Uses C-level SIMD kernel `c_abs`.
        - Registers C-level backward function `c_abs_grad_op`.
    """

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
    """
    Elementwise negation.

    Forward:
        out = -a

    Backward:
        dL/da = -dL/dout

    Notes:
        - Uses C-level SIMD kernel `c_neg`.
        - Registers C-level backward function `c_neg_grad_op`.
    """

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
    """
    Reduction: sum over a given axis.

    Forward:
        out = Σ a along `axis`

    Backward:
        dL/da = broadcast(dL/dout, shape of a)

    Notes:
        - Uses C-level kernel `c_sum`.
        - Backward not yet implemented (TODO: `sum_grad_op`).
    """

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
        c_sum_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)


class Mean(Function):
    """
    Reduction: mean over a given axis.

    Forward:
        out = mean(a, axis)

    Backward:
        dL/da = broadcast((1/N) * dL/dout, shape of a)

    Notes:
        - Uses C-level kernel `c_mean`.
    """

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
        c_mean_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)


class Max(Function):
    """
    Reduction: maximum over a given axis.

    Forward:
        out = max(a, axis)

    Backward:
        dL/da = dL/dout if a is the max element, else 0

    Notes:
        - Uses C-level kernel `c_max`.
        - Backward not yet implemented (TODO: `max_grad_op`).
    """

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
        c_max_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)


# --- Movement Operations ---


class View(Function):
    """
    Reshape tensor without copying data.

    Forward:
        out = view(a, shape)

    Backward:
        dL/da = reshape(dL/dout, shape of a)

    Notes:
        - Uses C-level kernel `c_view`.
        - Backward not yet implemented (TODO: `view_grad_op`).
    """

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
    """
    Insert a dimension of size 1 at the given axis.

    Forward:
        out = unsqueeze(a, dim)

    Backward:
        dL/da = squeeze(dL/dout, dim)

    Notes:
        - Uses C-level kernel `c_unsqueeze`.
        - Backward not yet implemented (TODO: `unsqueeze_grad_op`).
    """

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
    """
    Remove a dimension of size 1 at the given axis.

    Forward:
        out = squeeze(a, dim)

    Backward:
        dL/da = unsqueeze(dL/dout, dim)

    Notes:
        - Uses C-level kernel `c_squeeze`.
        - Backward not yet implemented (TODO: `squeeze_grad_op`).
    """

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
    """
    Swap two tensor dimensions.

    Forward:
        out = transpose(a, n, m)

    Backward:
        dL/da = transpose(dL/dout, n, m)

    Notes:
        - Uses C-level kernel `c_transpose`.
        - Backward not yet implemented (TODO: `transpose_grad_op`).
    """

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
    """
    Expand tensor to a larger shape.

    Forward:
        out = expand(a, shape)

    Backward:
        dL/da = reduce(dL/dout, original shape of a)

    Notes:
        - Uses C-level kernel `c_expand`.
        - Backward not yet implemented (TODO: `expand_grad_op`).
    """

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

class Broadcast(Function):
    """
    Broadcast tensor to a larger shape.

    Forward:
        out = broadcast(a, shape)

    Backward:
        dL/da = reduce(dL/dout, original shape of a)

    Notes:
        - Uses C-level kernel `c_broadcast`.
    """

    lazy_op_class = LazyBroadcast

    def _get_backward_fn_type(self) -> Optional[BackwardFnType]:
        return None

    def _execute_forward(
        self, out_tensor: "Tensor", a: "Tensor", shape: list[int], ndim: int
    ) -> "Tensor":
        from ..core.tensor import Tensor

        c_broadcast(a._c_tensor, out_tensor._c_tensor, ndim, shape)

        return out_tensor

    def backward(
        self,
        out_tensor_ptr: ctypes.POINTER(CTensor),
        prev_tensor_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)),
        n_prev: int,
        extras: ctypes.c_void_p,
    ):
        pass  # TODO: Implement expand_grad_op

