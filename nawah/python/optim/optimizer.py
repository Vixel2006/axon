from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def step(self):
        raise NotImplementedError("Please choose an optimizer.")

    @abstractmethod
    def zero_grad(self):
        raise NotImplementedError("Please choose an optimizer.")
