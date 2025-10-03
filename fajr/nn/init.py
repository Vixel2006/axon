import math
from fajr.functions import uniform, randn
from fajr.core.tensor import Tensor

def xavier_uniform_(shape: tuple[int, ...], in_features: int, out_features: int) -> Tensor:
    bound = math.sqrt(6 / (in_features + out_features))
    return uniform(shape, low=-bound, high=bound)

def xavier_normal_(shape: tuple[int, ...], in_features: int, out_features: int) -> Tensor:
    std = math.sqrt(2 / (in_features + out_features))
    return randn(shape) * std

def kaiming_uniform_(shape: tuple[int, ...], in_features: int) -> Tensor:
    bound = math.sqrt(6 / in_features)
    return uniform(shape, low=-bound, high=bound)

def kaiming_normal_(shape: tuple[int, ...], in_features: int) -> Tensor:
    std = math.sqrt(2 / in_features)
    return randn(shape) * std
