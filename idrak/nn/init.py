import math
from idrak.functions import uniform
from idrak.core.tensor import Tensor

def xavier_uniform_(shape: tuple[int, ...], in_features: int, out_features: int) -> Tensor:
    bound = math.sqrt(6 / in_features + out_features)
    return uniform(shape, low=-bound, high=bound)

def xavier_normal_(shape: tuple[int, ...], in_features: int, out_features: int) -> Tensor:
    lower_bound = 0
    upper_bound = 2 / (in_features + out_features)
    return uniform(shape, low=lower_bound, high=upper_bound)

def kaiming_uniform_(shape: tuple[int, ...], in_features: int) -> Tensor:
    bound = math.sqrt(6 / in_features)
    return uniform(shape, low=-bound, high=bound)

def kaiming_normal_(shape: tuple[int, ...], in_features: int) -> Tensor:
    lower_bound = 0
    upper_bound = 2 / in_features
    return uniform(shape, low=lower_bound, high=upper_bound)
