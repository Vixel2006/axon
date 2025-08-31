from typing import Union, Tuple
from py.elnawah_bindings.ctypes_definitions import CTensor
from py.ops.lazy_base import LazyOp
import math

class LazyAdd(LazyOp):
    """
    Lazy addition shape inference.

    Rules:
        - If `b` is a Tensor/CTensor: shapes must match exactly.
        - If `b` is a scalar: shape of `a` is preserved.

    Returns:
        Tuple[int, ...]: Same shape as `a`.
    """

    def calculate_output_shape(self, a: "Tensor", b: "Tensor") -> Tuple[int, ...]:
        from py.core.tensor import Tensor

        if isinstance(b, (Tensor, CTensor)):
            if a.shape != b.shape:
                raise RuntimeError(
                    f"Can't add tensors with shapes {a.shape} and {b.shape}"
                )
        return a.shape


class LazySub(LazyOp):
    """
    Lazy subtraction shape inference.

    Rules:
        - If `b` is a Tensor/CTensor: shapes must match exactly.
        - If `b` is a scalar: shape of `a` is preserved.

    Returns:
        Tuple[int, ...]: Same shape as `a`.
    """

    def calculate_output_shape(self, a: "Tensor", b: "Tensor") -> Tuple[int, ...]:
        from py.core.tensor import Tensor

        if isinstance(b, (Tensor, CTensor)):
            if a.shape != b.shape:
                raise RuntimeError(
                    f"Can't subtract tensors with shapes {a.shape} and {b.shape}"
                )
            return a.shape
        else:
            return a.shape


class LazyRSub(LazyOp):
    """
    Lazy right-subtraction shape inference (scalar - tensor).

    Rules:
        - Always returns the shape of `a`.

    Returns:
        Tuple[int, ...]: Same shape as `a`.
    """

    def calculate_output_shape(self, a: "Tensor", b: float) -> Tuple[int, ...]:
        return a.shape


class LazyMul(LazyOp):
    """
    Lazy multiplication shape inference.

    Rules:
        - If `b` is a Tensor/CTensor: shapes must match exactly.
        - If `b` is a scalar: shape of `a` is preserved.

    Returns:
        Tuple[int, ...]: Same shape as `a`.
    """

    def calculate_output_shape(self, a: "Tensor", b: "Tensor") -> Tuple[int, ...]:
        from py.core.tensor import Tensor

        if isinstance(b, (Tensor, CTensor)):
            if a.shape != b.shape:
                raise RuntimeError(
                    f"Can't multiply tensors with shapes {a.shape} and {b.shape}"
                )
            return a.shape
        else:
            return a.shape


class LazyDiv(LazyOp):
    """
    Lazy division shape inference.

    Rules:
        - If `b` is a Tensor/CTensor: shapes must match exactly.
        - If `b` is a scalar: shape of `a` is preserved.

    Returns:
        Tuple[int, ...]: Same shape as `a`.
    """

    def calculate_output_shape(self, a: "Tensor", b: "Tensor") -> Tuple[int, ...]:
        from py.core.tensor import Tensor

        if isinstance(b, (Tensor, CTensor)):
            if a.shape != b.shape:
                raise RuntimeError(
                    f"Can't divide tensors with shapes {a.shape} and {b.shape}"
                )
            return a.shape
        else:
            return a.shape


class LazyRDiv(LazyOp):
    """
    Lazy right-division shape inference (scalar / tensor).

    Rules:
        - Always returns the shape of `a`.

    Returns:
        Tuple[int, ...]: Same shape as `a`.
    """

    def calculate_output_shape(self, a: "Tensor", b: float) -> Tuple[int, ...]:
        return a.shape


class LazyMatMul(LazyOp):
    """
    Lazy matrix multiplication shape inference.

    Rules:
        - Requires `a.shape[-1] == b.shape[-2]`.
        - Output shape = `a.shape[:-1] + [b.shape[-1]]`.

    Returns:
        Tuple[int, ...]: Resulting shape after matmul.
    Raises:
        RuntimeError: If inner dimensions are incompatible.
    """

    def calculate_output_shape(self, a: "Tensor", b: "Tensor") -> Tuple[int, ...]:
        if a.shape[-1] != b.shape[-2]:
            raise RuntimeError(
                f"Can't multiply tensors with shapes {a.shape} and {b.shape}"
            )
        output_shape = list(a.shape[:-1]) + [b.shape[-1]]
        return tuple(output_shape)


class LazyConv2d(LazyOp):
    def calculate_output_shape(
        self,
        t: "Tensor",
        kernel: "Tensor",
        kernel_size: tuple[int, int],
        stride: Union[tuple[int, int], int] = (1, 1),
        padding: int = 0,
    ) -> Tuple[int, ...]:
        output_shape = [t.shape[0], kernel.shape[0]]

        if isinstance(stride, int):
            stride = (stride, stride)

        Wout = math.floor((t.shape[-1] - kernel_size[-1] + 2*padding  + 1) / stride[1])
        Hout = math.floor((t.shape[-2] - kernel_size[-2] + 2*padding + 1) / stride[0])

        output_shape.extend([Hout, Wout])

        return tuple(output_shape)
