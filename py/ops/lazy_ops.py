from typing import Any, Tuple
from ..elnawah_bindings.ctypes_definitions import CTensor


class LazyOp:
    """
    Abstract base class for lazy tensor operations.

    Responsibilities:
        - Defines the interface for computing output shapes
          without executing the actual operation.

    Methods:
        - calculate_output_shape(...):
            Must be implemented by subclasses to infer the resulting
            tensor shape from input tensor(s) and parameters.
    """

    def calculate_output_shape(self, *args: Any, **kwargs: Any) -> Tuple[int, ...]:
        raise NotImplementedError


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
        else:
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


class LazyView(LazyOp):
    """
    Lazy view/reshape shape inference.

    Rules:
        - Input `numel` must match `numel` of target shape.
        - Reshape does not alter number of elements.

    Returns:
        Tuple[int, ...]: New shape.

    Raises:
        RuntimeError: If element counts mismatch.
    """

    def calculate_output_shape(self, a: "Tensor", shape: list) -> Tuple[int, ...]:
        from py.core.tensor import Tensor

        if a.numel() != Tensor.safe_c_numel(shape, len(shape)):
            raise RuntimeError(
                f"Unable to operate view as {a.numel()} != {Tensor.safe_c_numel(shape, len(shape))}"
            )
        return tuple(shape)


class LazyUnsqueeze(LazyOp):
    """
    Lazy unsqueeze shape inference.

    Args:
        dim (int): Dimension index to insert a singleton.

    Rules:
        - Inserts a new dimension of size 1 at `dim`.

    Returns:
        Tuple[int, ...]: Shape with one extra dimension.

    Raises:
        ValueError: If `dim` > ndim of input tensor.
    """

    def calculate_output_shape(self, a: "Tensor", dim: int = 0) -> Tuple[int, ...]:
        if dim < 0:
            dim = a.ndim + dim + 1
        if dim > a.ndim:
            raise ValueError(f"Can't unsqueeze dim {dim}.")
        new_shape = list(a.shape)
        new_shape.insert(dim, 1)
        return tuple(new_shape)


class LazySqueeze(LazyOp):
    """
    Lazy squeeze shape inference.

    Args:
        dim (int): Dimension index to remove.

    Rules:
        - Removes dimension if its size == 1.
        - Negative dims supported.

    Returns:
        Tuple[int, ...]: Shape with one less dimension.

    Raises:
        ValueError: If `dim` out of range.
        RuntimeError: If target dim is not size 1.
    """

    def calculate_output_shape(self, a: "Tensor", dim: int = 0) -> Tuple[int, ...]:
        if dim < 0:
            dim = a.ndim + dim
        if dim >= a.ndim:
            raise ValueError(f"Can't squeeze dim {dim} as it doesn't exist.")
        if a.shape[dim] != 1:
            raise RuntimeError(
                f"Tensors can be squeezed only on shape[dim] = 1, dim = {a.shape[dim]}"
            )
        new_shape = list(a.shape)
        new_shape.pop(dim)
        return tuple(new_shape)


class LazyTranspose(LazyOp):
    """
    Lazy transpose shape inference.

    Args:
        n (int): First axis.
        m (int): Second axis.

    Rules:
        - Swaps dimensions `n` and `m`.
        - Negative indices allowed.

    Returns:
        Tuple[int, ...]: Shape with axes swapped.

    Raises:
        ValueError: If axes are invalid.
    """

    def calculate_output_shape(
        self, a: "Tensor", n: int = -2, m: int = -1
    ) -> Tuple[int, ...]:
        if n < 0:
            n = a.ndim + n
        if m < 0:
            m = a.ndim + m
        if n >= a.ndim or m >= a.ndim:
            raise ValueError(f"Can't transpose around non-existing axes {n},{m}")
        new_shape = list(a.shape)
        new_shape[n], new_shape[m] = new_shape[m], new_shape[n]
        return tuple(new_shape)


class LazyExpand(LazyOp):
    """
    Lazy expand shape inference.

    Args:
        shape (list[int]): Target shape.

    Rules:
        - Input must have same ndim as `shape`.
        - Dimension can be expanded if it's 1.
        - Expanding from !=1 to a new size is invalid.

    Returns:
        Tuple[int, ...]: Expanded shape.

    Raises:
        ValueError: If ndim mismatch.
        RuntimeError: If expansion rule is violated.
    """

    def calculate_output_shape(self, a: "Tensor", shape: list[int]) -> Tuple[int, ...]:
        if a.ndim != len(shape):
            raise ValueError(f"expand() error: Dimensionality mismatch")
        for i in range(a.ndim):
            if a.shape[i] != shape[i] and a.shape[i] != 1:
                raise RuntimeError(
                    f"expand() error: Can't expand dim {i} from {a.shape[i]} to {shape[i]}, Only dims of 1 can be expanded."
                )
        return tuple(shape)

class LazyBroadcast(LazyOp):
    """
    Lazy broadcast shape inference.

    Args:
        shape (list[int]): Target shape.

    Rules:
        - Input must have same ndim as `shape`.
        - Dimension can be expanded if it's 1.
        - Expanding from !=1 to a new size is invalid.

    Returns:
        Tuple[int, ...]: broadcasted shape.

    Raises:
        RuntimeError: If expansion rule is violated.
    """

    def calculate_output_shape(self, a: "Tensor", shape: list[int], ndim: int) -> Tuple[int, ...]:
        a_shape = list(a.shape)
        target_shape = list(shape)

        # The target shape determines the output dimensionality
        output_ndim = len(target_shape)

        # Pad a_shape with leading ones to match the target_shape's dimensionality
        if len(a_shape) > output_ndim:
            raise RuntimeError(
                f"broadcast() error: Cannot broadcast tensor with shape {a.shape} to target shape {target_shape}. "
                f"Tensor has more dimensions than the target shape."
            )
        
        padded_a_shape = [1] * (output_ndim - len(a_shape)) + a_shape

        # Check compatibility from right to left
        for i in range(1, output_ndim + 1): # Iterate from last dimension to first
            dim_a = padded_a_shape[-i]
            dim_target = target_shape[-i]

            if dim_a == dim_target:
                continue # Compatible
            elif dim_a == 1:
                continue # Compatible, 1 can broadcast to any size
            elif dim_target == 1:
                # If target_dim is 1, then a_dim must also be 1 for compatibility in broadcast_to
                if dim_a != 1:
                    raise RuntimeError(
                        f"broadcast() error: Cannot broadcast dimension {dim_a} to {dim_target} at dim {-i}. "
                        f"Target dimension is 1, but input dimension is not 1."
                    )
            else:
                raise RuntimeError(
                    f"broadcast() error: Dimensions {dim_a} and {dim_target} are not compatible for broadcasting at dim {-i}."
                )
        
        return tuple(target_shape)

