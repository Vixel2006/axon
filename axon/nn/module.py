from abc import ABC, abstractmethod
from axon.core.tensor import Tensor
from axon.axon_bindings.c_wrapper_functions import c_tfree
from typing import Any, Iterable

class Module(ABC):
    @abstractmethod
    def forward(self, x, *args, **kwargs):
        pass

    def __call__(self, x, *args: Any, **kwds: Any) -> Any:
        return self.forward(x, *args, **kwds)

    def __rshift__(self, other):
        from .pipeline import Pipeline
        if isinstance(other, Module):
            return Pipeline(self, other)
        elif isinstance(other, Pipeline):
            new_pipeline = Pipeline(self)
            new_pipeline.layers.extend(other.layers)
            return new_pipeline
        else:
            return NotImplemented

    @property
    def params(self):
        params = []
        for elem in self.__dict__.values():
            if isinstance(elem, Tensor) and elem.requires_grad:
                params.append(elem)
            elif isinstance(elem, Module):
                params.extend(elem.params)
        return params

    @property
    def buffers(self):
        buffers = []
        for elem in self.__dict__.values():
            if isinstance(elem, Tensor) and not elem.requires_grad:
                buffers.append(elem)
            elif isinstance(elem, Module):
                buffers.extend(elem.buffers)
        return buffers

    def freeze(self):
        for param in self.params:
            if param.c_tensor_ptr.contents.grad:
                c_tfree(param.grad.c_tensor_ptr)
                param.c_tensor_ptr.contents.grad = None
            param.requires_grad = False

    @abstractmethod
    def reset_parameters(self):
        pass
