import cnawah as nw

class Tensor(nw.Tensor):
    def __rshift__(self, fn):
        return fn(self)

    def add(self, other):
        return self + other
