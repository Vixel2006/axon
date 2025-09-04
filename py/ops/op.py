from __future__ import annotations
from typing import Any
from abc import ABC, abstractmethod
from py.idrak_bindings.ctypes_definitions import CTensor

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
        from py.core.node import Node
        from py.core.tensor import Tensor

        out_shape = cls.calc_out_shape(*args, **kwargs)

        requires_grad = any(isinstance(arg, CTensor) and arg.requires_grad for arg in args)

        out = Tensor(shape=out_shape, requires_grad=requires_grad)

        in_tensors = [arg for arg in args if isinstance(arg, CTensor)]

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
