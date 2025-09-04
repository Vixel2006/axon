from py.functions import *
from py.core.tensor import Tensor

def mse(pred: Tensor, truth: Tensor, reduction: str = "mean"):
    if reduction == "mean":
        return mean((pred - truth) ** 2)
    elif reduction == "sum":
        return sum((pred - truth) ** 2)

if __name__ == "__main__":
    a = Tensor((2,2), [[1,2], [3, 4]])
    b = Tensor((2,2), [[2,4], [4, 6]])

    c = mse(b, a)

    c.backward()

    print(c)
    print(a.grad)
    print(b.grad)
