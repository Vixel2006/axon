from abc import ABC, abstractmethod
from idrak.core.tensor import Tensor
from typing import Any, Iterable

class Module(ABC):
    @abstractmethod
    def forward(self, x, *args, **kwargs):
        pass

    def __call__(self, x, *args: Any, **kwds: Any) -> Any:
        return self.forward(x, *args, **kwds)

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
