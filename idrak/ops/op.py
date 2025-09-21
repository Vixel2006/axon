from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Dict, Optional

class LazyOp(ABC):
    @abstractmethod
    def calc_out_shape(self, *args, **kwargs) -> tuple[int, ...]:
        pass

    @abstractmethod
    def create_ctx_struct(self, *args, **kwargs) -> tuple[Dict[str, Any], Any]:
        pass

    @abstractmethod
    def forward(self, out: "Tensor", *args, **kwargs): # Added 'out' as first positional arg
        pass

    @abstractmethod
    def backward(self, out_ptr: ctypes.POINTER("CTensor"), prev_ptrs: ctypes.POINTER(ctypes.POINTER("CTensor")), n_prev: int, extras: Any):
        pass

    @classmethod
    def create_node(cls, *args, **kwargs):
        from idrak.core.tensor import Tensor # Ensure Tensor is imported
        from idrak.core.buffer import LazyBuffer # Ensure LazyBuffer is imported
        from idrak.idrak_bindings.ctypes_definitions import CTensor # Ensure CTensor is imported

        # Calculate output shape
        out_shape = cls.calc_out_shape(*args, **kwargs)

        # Process inputs and determine if gradients are needed
        processed_inputs_for_node = []
        requires_grad_flag = False

        # This loop correctly identifies actual Tensor dependencies
        for arg in args:
            if isinstance(arg, Tensor):
                processed_inputs_for_node.append(arg)
                if arg.requires_grad:
                    requires_grad_flag = True
            elif isinstance(arg, CTensor): # If CTensor can be a direct input
                temp_tensor = Tensor._wrap_c_tensor_ptr(arg) # Use wrapper
                processed_inputs_for_node.append(temp_tensor)
                if temp_tensor.requires_grad:
                    requires_grad_flag = True
            elif isinstance(arg, (list, tuple)):
                for item in arg:
                    if isinstance(item, Tensor):
                        processed_inputs_for_node.append(item)
                        if item.requires_grad:
                            requires_grad_flag = True
                    elif isinstance(item, CTensor):
                        temp_tensor = Tensor._wrap_c_tensor_ptr(item) # Use wrapper
                        processed_inputs_for_node.append(temp_tensor)
                        if temp_tensor.requires_grad:
                            requires_grad_flag = True
        
        # Determine if the *output* tensor should require gradients
        # Usually, this is explicitly passed as a kwarg, or inferred from inputs
        output_requires_grad = kwargs.pop('requires_grad', requires_grad_flag) # Allow explicit override

        # Create output tensor
        out = Tensor(shape=out_shape, requires_grad=output_requires_grad)

        # Create context: now returns (forward_kwargs, backward_ctx)
        forward_kwargs, backward_ctx = cls.create_ctx_struct(*args, **kwargs)

        # Create lazy buffer - this is the only computation tracking mechanism
        # For CreationOps, processed_inputs_for_node will be empty, which is correct.
        lazy_buffer = LazyBuffer(out, cls(), processed_inputs_for_node, forward_kwargs, backward_ctx)
        out._lazy_buffer = lazy_buffer

        return out
