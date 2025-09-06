from abc import ABC, abstractmethod
from idrak.core.tensor import Tensor
from typing import Any

class Module(ABC):
    @abstractmethod
    def forward(self, x, *args, **kwargs):
        pass

    def __call__(self, x, *args: Any, **kwds: Any) -> Any:
        return self.forward(x, *args, **kwds)

    @property
    def params(self):
        params_list = []
        for _, value in self.__dict__.items():
            if isinstance(value, Tensor):
                if value.requires_grad:
                    params_list.append(value)
            elif isinstance(value, Module):
                params_list.extend(value.params)
        return params_list

    @property
    def buffers(self):
        buffers_list = []
        for _, value in self.__dict__.items():
            if isinstance(value, Tensor):
                if not value.requires_grad:
                    buffers_list.append(value)
            elif isinstance(value, Module):
                buffers_list.extend(value.buffers)
        return buffers_list
