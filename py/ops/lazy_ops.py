from typing import Any, Tuple
from ..elnawah_bindings.ctypes_definitions import CTensor


class LazyOp:
    def calculate_output_shape(self, *args: Any, **kwargs: Any) -> Tuple[int, ...]:
        raise NotImplementedError


class LazyAdd(LazyOp):
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
    def calculate_output_shape(self, a: "Tensor", b: float) -> Tuple[int, ...]:
        return a.shape


class LazyMul(LazyOp):
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
    def calculate_output_shape(self, a: "Tensor", b: float) -> Tuple[int, ...]:
        return a.shape


class LazyMatMul(LazyOp):
    def calculate_output_shape(self, a: "Tensor", b: "Tensor") -> Tuple[int, ...]:
        if a.shape[-1] != b.shape[-2]:
            raise RuntimeError(
                f"Can't multiply tensors with shapes {a.shape} and {b.shape}"
            )
        output_shape = list(a.shape[:-1]) + [b.shape[-1]]
        return tuple(output_shape)


class LazyReLU(LazyOp):
    def calculate_output_shape(self, a: "Tensor") -> Tuple[int, ...]:
        return a.shape


class LazyLog(LazyOp):
    def calculate_output_shape(self, a: "Tensor") -> Tuple[int, ...]:
        return a.shape


class LazyExp(LazyOp):
    def calculate_output_shape(self, a: "Tensor") -> Tuple[int, ...]:
        return a.shape


class LazySoftmax(LazyOp):
    def calculate_output_shape(self, a: "Tensor") -> Tuple[int, ...]:
        return a.shape


class LazyAbs(LazyOp):
    def calculate_output_shape(self, a: "Tensor") -> Tuple[int, ...]:
        return a.shape


class LazyNeg(LazyOp):
    def calculate_output_shape(self, a: "Tensor") -> Tuple[int, ...]:
        return a.shape


class LazySum(LazyOp):
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
    def calculate_output_shape(self, a: "Tensor", shape: list) -> Tuple[int, ...]:
        from py.core.tensor import Tensor

        if a.numel() != Tensor.safe_c_numel(shape, len(shape)):
            raise RuntimeError(
                f"Unable to operate view as {a.numel()} != {Tensor.safe_c_numel(shape, len(shape))}"
            )
        return tuple(shape)


class LazyUnsqueeze(LazyOp):
    def calculate_output_shape(self, a: "Tensor", dim: int = 0) -> Tuple[int, ...]:
        if dim < 0:
            dim = a.ndim + dim + 1
        if dim > a.ndim:
            raise ValueError(f"Can't unsqueeze dim {dim}.")
        new_shape = list(a.shape)
        new_shape.insert(dim, 1)
        return tuple(new_shape)


class LazySqueeze(LazyOp):
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
    def calculate_output_shape(self, a: "Tensor", shape: list[int]) -> Tuple[int, ...]:
        if a.ndim != len(shape):
            raise ValueError(f"expand() error: Dimensionality mismatch")
        for i in range(a.ndim):
            if a.shape[i] != shape[i] and a.shape[i] != 1:
                raise RuntimeError(
                    f"expand() error: Can't expand dim {i} from {a.shape[i]} to {shape[i]}, Only dims of 1 can be expanded."
                )
        return tuple(shape)
