from __future__ import annotations
import ctypes
from typing import Any, Tuple, Optional, List
import sys

from py.elnawah_bindings.ctypes_definitions import CTensor, BackwardFnType
from py.elnawah_bindings.c_library_loader import tensor_lib


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
                forward_args=(output_tensor, *args),
                forward_kwargs=kwargs,
                backward_fn=backward_fn,
                extras=ctx.extras,
            )
        else:
            # For non-grad tensors, execute immediately
            result = ctx._execute_forward(output_tensor, *args, **kwargs)
            output_tensor = result

        return output_tensor