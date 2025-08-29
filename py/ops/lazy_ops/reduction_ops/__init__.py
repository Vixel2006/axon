from typing import Any, Tuple
from py.ops.lazy_base import LazyOp


class LazySum(LazyOp):
    """
    Lazy summation shape inference.

    Args:
        axis (int): Axis along which to reduce.
        keepdim (bool): Whether to keep reduced dimension.

    Rules:
        - If `keepdim=True`: dimension `axis` is replaced with 1.
        - If `keepdim=False`: dimension `axis` is removed.

    Returns:
        Tuple[int, ...]: Reduced shape.
    """

    def calculate_output_shape(
        self, a: "Tensor", axis: int = 0, keepdim: bool = True
    ) -> Tuple[int, ...]:
        if not keepdim:
            new_shape = list(a.shape)
            new_shape.pop(axis)
            return tuple(new_shape)
        else:
            new_shape = list(a.shape)
            new_shape[axis] = 1
            return tuple(new_shape)


class LazyMean(LazyOp):
    """
    Lazy mean shape inference.

    Args:
        axis (int): Axis along which to reduce.
        keepdim (bool): Whether to keep reduced dimension.

    Rules:
        - If `keepdim=True`: dimension `axis` is replaced with 1.
        - If `keepdim=False`: dimension `axis` is removed.

    Returns:
        Tuple[int, ...]: Reduced shape.
    """

    def calculate_output_shape(
        self, a: "Tensor", axis: int = 0, keepdim: bool = True
    ) -> Tuple[int, ...]:
        if not keepdim:
            new_shape = list(a.shape)
            new_shape.pop(axis)
            return tuple(new_shape)
        else:
            new_shape = list(a.shape)
            new_shape[axis] = 1
            return tuple(new_shape)


class LazyMax(LazyOp):
    """
    Lazy max reduction shape inference.

    Args:
        axis (int): Axis along which to reduce.
        keepdim (bool): Whether to keep reduced dimension.

    Rules:
        - If `keepdim=True`: dimension `axis` is replaced with 1.
        - If `keepdim=False`: dimension `axis` is removed.

    Returns:
        Tuple[int, ...]: Reduced shape.
    """

    def calculate_output_shape(
        self, a: "Tensor", axis: int = 0, keepdim: bool = True
    ) -> Tuple[int, ...]:
        if not keepdim:
            new_shape = list(a.shape)
            new_shape.pop(axis)
            return tuple(new_shape)
        else:
            new_shape = list(a.shape)
            new_shape[axis] = 1
            return tuple(new_shape)
