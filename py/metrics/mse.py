from py.functions import *
from py.core.tensor import Tensor

def mse(pred: Tensor, truth: Tensor, reduction: str = "mean"):
    if reduction == "mean":
        return mean(((pred - truth) ** 2))
    elif reduction == "sum":
        return sum(((pred - truth) ** 2))

if __name__ == "__main__":
    a = Tensor((3,2), [[1,2], [3,4], [3,4]])
    b = Tensor((3,2), [[2,4], [4, 6], [4, 6]])

    c = (b - a)
    d = c ** 2

    e = sum(d)


    print(f"Ouput Tensor: {e}")
    print(f"e grad: {e.grad}")
    print(f"d grad:{d.grad}")
    print(f"c grad: {c.grad}")
    print(f"b grad: {b.grad}")
    print(f"a grad: {a.grad}")

    e.backward()

    print(f"Ouput Tensor: {e}")
    print(f"e grad: {e.grad}")
    print(f"d grad:{d.grad}")
    print(f"c grad: {c.grad}")
    print(f"b grad: {b.grad}")
    print(f"a grad: {a.grad}")

