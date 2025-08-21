import cnawah as cnw
from ..api.pipeline import Pipe, Pipeable


class Tensor(cnw.Tensor):
    def __rshift__(self, other):
        if isinstance(other, Pipe):
            return other._execute(self)
        else:
            return other(self)

    def relu(self, leak=0):
        pass
