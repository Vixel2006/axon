from abc import ABC, abstractmethod

from ..core.tensor import Tensor


class Module(ABC):
    def __init__(self):
        self.training = True
        self._modules = {}

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward method is not implemented yet.")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self):
        self.training = True
        for module in self._modules.values():
            if module is not None:
                module.train()
        return self

    def eval(self):
        self.training = False
        for module in self._modules.values():
            if module is not None:
                module.eval()
        return self

    def parameters(self):
        params = []
        for value in self.__dict__.values():
            if isinstance(value, Tensor):
                params.append(value)

        for module in self._modules.values():
            if module is not None:
                params.extend(module.parameters())

        return list(dict.fromkeys(params))

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def apply(self, fn):
        fn(self)

        for module in self._modules.values():
            if module is not None:
                module.apply(fn)

        return self
