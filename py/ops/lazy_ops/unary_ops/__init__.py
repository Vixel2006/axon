from typing import Any, Tuple
from py.ops.lazy_base import LazyOp


class LazyReLU(LazyOp):
    """
    Lazy ReLU shape inference.

    Rules:
        - Output shape is identical to input shape.

    Returns:
        Tuple[int, ...]: Same shape as `a`.
    """

    def calculate_output_shape(self, a: "Tensor") -> Tuple[int, ...]:
        return a.shape


class LazyLog(LazyOp):
    """Lazy log shape inference. Shape is preserved."""

    def calculate_output_shape(self, a: "Tensor") -> Tuple[int, ...]:
        return a.shape


class LazyExp(LazyOp):
    """Lazy exponential shape inference. Shape is preserved."""

    def calculate_output_shape(self, a: "Tensor") -> Tuple[int, ...]:
        return a.shape


class LazySoftmax(LazyOp):
    """Lazy softmax shape inference. Shape is preserved."""

    def calculate_output_shape(self, a: "Tensor") -> Tuple[int, ...]:
        return a.shape


class LazyAbs(LazyOp):
    """Lazy absolute-value shape inference. Shape is preserved."""

    def calculate_output_shape(self, a: "Tensor") -> Tuple[int, ...]:
        return a.shape


class LazyNeg(LazyOp):
    """Lazy negation shape inference. Shape is preserved."""

    def calculate_output_shape(self, a: "Tensor") -> Tuple[int, ...]:
        return a.shape
