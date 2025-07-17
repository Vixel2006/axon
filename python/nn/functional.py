from functools import wraps

from ..core import Tensor


def relu(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        return result.relu()

    return wrapper


def flatten(x):
    if not isinstance(x, Tensor):
        raise TypeError(f"flatten expects a Tensor, but got {type(x).__name__}")
    return x.view(x.shape[0], -1)
