from __future__ import annotations
from typing import Any
from abc import ABC, abstractmethod
from idrak.idrak_bindings.ctypes_definitions import CTensor

class LazyOp(ABC):
    @abstractmethod
    def calc_out_shape(self, *args, **kwargs) -> tuple[int, ...]:
        pass
    
    @abstractmethod
    def create_ctx_struct(self, *args, **kwargs) -> Any:
        pass
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass
    
    @classmethod
    def create_node(cls, *args, **kwargs):
        from idrak.core.node import Node
        from idrak.core.tensor import Tensor

        out_shape = cls.calc_out_shape(*args, **kwargs)

        # Collect all Tensor inputs for requires_grad and input_tensors
        processed_inputs_for_node = []
        requires_grad_flag = False

        for arg in args:
            if isinstance(arg, Tensor):
                processed_inputs_for_node.append(arg)
                if arg.requires_grad:
                    requires_grad_flag = True
            elif isinstance(arg, CTensor):
                # If a raw CTensor is passed, it needs to be wrapped in a Tensor object.
                # This assumes Tensor can be initialized with a c_tensor argument.
                temp_tensor = Tensor(c_tensor=arg)
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
                        temp_tensor = Tensor(c_tensor=item)
                        processed_inputs_for_node.append(temp_tensor)
                        if temp_tensor.requires_grad:
                            requires_grad_flag = True

        out = Tensor(shape=out_shape, requires_grad=requires_grad_flag)

        in_tensors = processed_inputs_for_node

        extras = cls.create_ctx_struct(*args, **kwargs)

        out._node = Node(
            out_tensor=out,
            input_tensors=in_tensors,
            forward_fn=cls.forward,
            forward_args=(out, *args),
            forward_kwargs=kwargs,
            backward_fn=cls.backward,
            extras=extras
        )
        return out
