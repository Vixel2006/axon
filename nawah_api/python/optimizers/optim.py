import cnawah as nw
from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def step(self):
        raise NotImplementedError("Not implemented yet.")

    def zero_grad(self):
        raise NotImplementedError("Not implemented yet.")
