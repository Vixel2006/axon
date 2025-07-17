import numpy as np

from .tensor import Tensor


def _calculate_fan_in_and_fan_out(tensor: Tensor):
    dimensions = tensor.ndim
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can't be computed for tensor with fewer than 2 dimensions"
        )

    if dimensions == 2:
        fan_in, fan_out = tensor.shape[0], tensor.shape[1]
    else:
        receptive_field_size = np.prod(tensor.shape[2:])
        fan_in = tensor.shape[1] * receptive_field_size
        fan_out = tensor.shape[0] * receptive_field_size

    return fan_in, fan_out


def kaiming_uniform_(
    tensor: Tensor, a: float = 0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"
):
    if not tensor.requires_grad:
        return tensor

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)

    if mode == "fan_in":
        fan = fan_in
    else:
        fan = fan_out

    if nonlinearity == "relu" or nonlinearity == "leaky_relu":
        gain = np.sqrt(2.0 / (1 + a**2))
    elif nonlinearity == "tanh":
        gain = 5.0 / 3.0
    else:
        gain = 1.0

    std = gain / np.sqrt(fan)
    bound = np.sqrt(3.0) * std

    tensor.data[:] = np.random.uniform(-bound, bound, size=tensor.shape)
    return tensor


def xavier_normal_(tensor: Tensor, gain: float = 1.0):
    if not tensor.requires_grad:
        return tensor

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))

    tensor.data[:] = np.random.normal(0, std, size=tensor.shape)
    return tensor


def zeros_(tensor: Tensor):
    tensor.data.fill(0)
    return tensor


def ones_(tensor: Tensor):
    tensor.data.fill(1)
    return tensor
