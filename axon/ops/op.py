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
    def forward(self, out: "Tensor", *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, out_ptr: ctypes.POINTER("CTensor"), prev_ptrs: ctypes.POINTER(ctypes.POINTER("CTensor")), n_prev: int, extras: Any):
        pass

    @classmethod
    def create_node(cls, *args, **kwargs):
        from axon.core.tensor import Tensor
        from axon.core.buffer import LazyBuffer
        from axon.axon_bindings.ctypes_definitions import CTensor

        out_shape = cls.calc_out_shape(*args, **kwargs)

        processed_inputs_for_node = []
        requires_grad_flag = False

        for arg in args:
            if isinstance(arg, Tensor):
                processed_inputs_for_node.append(arg)
                if arg.requires_grad:
                    requires_grad_flag = True
            elif isinstance(arg, (list, tuple)):
                for item in arg:
                    if isinstance(item, Tensor):
                        processed_inputs_for_node.append(item)
                        if item.requires_grad:
                            requires_grad_flag = True
        
        output_requires_grad = kwargs.pop('requires_grad', requires_grad_flag)

        out = Tensor(shape=out_shape, requires_grad=output_requires_grad)

        forward_kwargs, backward_ctx = cls.create_ctx_struct(*args, **kwargs)

        lazy_buffer = LazyBuffer(out, cls(), processed_inputs_for_node, forward_kwargs, backward_ctx)
        out._lazy_buffer = lazy_buffer

        return out
