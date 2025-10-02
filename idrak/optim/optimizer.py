from abc import ABC, abstractmethod
from idrak.core.tensor import Tensor

class Optimizer(ABC):
    @abstractmethod
    def step(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def zero_grad(self):
        raise NotImplementedError
