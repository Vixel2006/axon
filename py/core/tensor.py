from __future__ import annotations

from ..elnawah_bindings.c_wrapper_functions import (
    c_malloc_tensor_empty,
    c_malloc_tensor_shape,
    c_malloc_tensor_full,
    c_free_tensor,
    c_numel,
    c_compute_strides,
    c_set_ones_grad,
)
from ..elnawah_bindings.ctypes_definitions import CTensor
from ..elnawah_bindings.c_library_loader import tensor_lib

from .node import Node
from py.ops.functions import (
    Add,
    Sub,
    RSub,
    Mul,
    Div,
    RDiv,
    MatMul,
    ReLU,
    Log,
    Exp,
    Softmax,
    Abs,
    Neg,
    Sum,
    Mean,
    Max,
    View,
    Unsqueeze,
    Squeeze,
    Transpose,
    Expand,
)

import numpy as np
import ctypes
from typing import Union


def _flatten_list(nested_list):
    if not isinstance(nested_list, (list, tuple)):
        return [nested_list]
    flattened = []
    for item in nested_list:
        if isinstance(item, (list, tuple)):
            flattened.extend(_flatten_list(item))
        else:
            flattened.append(item)
    return flattened


class Tensor(CTensor):
    def __init__(self, shape=None, data=None, requires_grad=True, _c_tensor_ptr=None):
        if _c_tensor_ptr is not None:
            self._c_tensor = _c_tensor_ptr
            if self._c_tensor and self._c_tensor.contents:
                if self._c_tensor.contents.ndim == 0:
                    self._shape = ()
                else:
                    self._shape = [
                        self._c_tensor.contents.shape[i]
                        for i in range(self._c_tensor.contents.ndim)
                    ]
            else:
                self._shape = None
            return

        self._c_tensor = None
        self._node = Node(
            out_tensor=self,
            input_tensors=[],
            forward_fn=None,
            forward_args=[],
            forward_kwargs={},
            backward_fn=None,
        )

        if shape is None and data is None:
            self._c_tensor = c_malloc_tensor_empty()
        elif shape is not None and data is None:
            ndim = len(shape)
            self._shape = shape
            self._c_tensor = c_malloc_tensor_shape(shape, ndim, requires_grad)
        elif shape is not None and data is not None:
            ndim = len(shape)
            self._shape = shape
            strides = c_compute_strides(shape, ndim)
            c_data = _flatten_list(data)
            self._c_tensor = c_malloc_tensor_full(
                shape, ndim, strides, c_data, requires_grad, None
            )
        else:
            raise ValueError("Invalid Tensor init arguments")

    @property
    def shape(self):
        return self._shape

    @property
    def data(self) -> np.ndarray:
        if not self._c_tensor or not self._c_tensor.contents:
            raise ValueError("Invalid tensor: NULL pointer")

        t = self._c_tensor.contents

        if t.ndim == 0:
            if not t.data:
                raise ValueError("Invalid tensor: NULL data pointer")
            return np.array(t.data[0], dtype=np.float32)

        if t.ndim < 0:
            raise ValueError(f"Invalid tensor: negative ndim {t.ndim}")

        if not t.shape:
            raise ValueError(
                "Invalid tensor: NULL shape pointer for multi-dimensional tensor"
            )

        if not t.data:
            raise ValueError("Invalid tensor: NULL data pointer")

        try:
            shape = (
                self._shape
                if self._shape is not None
                else [t.shape[i] for i in range(t.ndim)]
            )

            n = int(np.prod(shape)) if shape else 0

            if n <= 0:
                raise ValueError(f"Invalid tensor size: {n}")

            flat_data = []
            for i in range(n):
                try:
                    flat_data.append(t.data[i])
                except (IndexError, ValueError, ctypes.ArgumentError) as e:
                    raise ValueError(f"Failed to access tensor data at index {i}: {e}")

            np_array = np.array(flat_data, dtype=np.float32)
            return np_array.reshape(shape)

        except Exception as e:
            raise ValueError(f"Failed to convert tensor data to numpy array: {e}")

    @property
    def grad(self) -> np.ndarray:
        if not self._c_tensor or not self._c_tensor.contents:
            raise ValueError("Invalid tensor: NULL pointer")

        t = self._c_tensor.contents

        if not t.requires_grad or not t.grad:
            return None

        if t.ndim == 0:
            return np.array(t.grad[0], dtype=np.float32)

        try:
            n = int(np.prod(self._shape)) if self._shape else 0

            if n <= 0:
                raise ValueError(f"Invalid tensor size: {n}")

            flat_grad = []
            for i in range(n):
                try:
                    flat_grad.append(t.grad[i])
                except (IndexError, ValueError, ctypes.ArgumentError) as e:
                    raise ValueError(
                        f"Failed to access tensor gradient at index {i}: {e}"
                    )

            np_array = np.array(flat_grad, dtype=np.float32)
            return np_array.reshape(self._shape)

        except Exception as e:
            raise ValueError(f"Failed to convert tensor gradient to numpy array: {e}")

    @property
    def requires_grad(self) -> bool:
        if not self._c_tensor or not self._c_tensor.contents:
            return False
        return bool(self._c_tensor.contents.requires_grad)

    @property
    def ndim(self) -> int:
        if not self._c_tensor or not self._c_tensor.contents:
            return 0
        return self._c_tensor.contents.ndim

    def __str__(self) -> str:
        try:
            data_str = str(self.data)
            grad_info = f", requires_grad={self.requires_grad}"
            if self.requires_grad and self.grad is not None:
                grad_info += f", has_grad=True"
            return f"Tensor({data_str}{grad_info})"
        except Exception as e:
            return f"Tensor(invalid: {e})"

    def __repr__(self) -> str:
        return self.__str__()

    def __add__(self, other: Union[Tensor, float]) -> Tensor:
        return Add.apply(self, other)

    def __iadd__(self, other: Union[Tensor, float]) -> Tensor:
        return self + other

    def __radd__(self, other: float) -> Tensor:
        return self + other

    def __sub__(self, other: Union[Tensor, float]) -> Tensor:
        return Sub.apply(self, other)

    def __isub__(self, other: Union[Tensor, float]) -> Tensor:
        return self - other

    def __rsub__(self, other: float) -> Tensor:
        return RSub.apply(self, other)

    def __mul__(self, other: Union[Tensor, float]) -> Tensor:
        return Mul.apply(self, other)

    def __imul__(self, other: Union[Tensor, float]) -> Tensor:
        return self * other

    def __rmul__(self, other: float) -> Tensor:
        return self * other

    def __truediv__(self, other: Union[Tensor, float]) -> Tensor:
        return Div.apply(self, other)

    def __itruediv__(self, other: Union[Tensor, float]) -> Tensor:
        return self / other

    def __rtruediv__(self, other: float) -> Tensor:
        return RDiv.apply(self, other)

    def __matmul__(self, other: Tensor) -> Tensor:
        return MatMul.apply(self, other)

    def __neg__(self) -> Tensor:
        return Neg.apply(self)

    def sum(self, axis: int = 0, keepdim: bool = True):
        return Sum.apply(self, axis, keepdim)

    def mean(self, axis: int = 0, keepdim: bool = True):
        return Mean.apply(self, axis, keepdim)

    def max(self, axis: int = 0, keepdim: bool = True):
        return Max.apply(self, axis, keepdim)

    def validate(self) -> bool:
        try:
            if not self._c_tensor or not self._c_tensor.contents:
                return False

            t = self._c_tensor.contents

            if t.ndim < 0:
                return False

            if t.ndim == 0:
                return t.data is not None

            if not t.shape or not t.data:
                return False

            for i in range(t.ndim):
                if t.shape[i] <= 0:
                    return False

            if t.requires_grad and not t.grad:
                return False

            return True

        except Exception:
            return False

    def view(self, shape: list) -> Tensor:
        return View.apply(self, shape)

    def unsqueeze(self, dim: int = 0) -> Tensor:
        return Unsqueeze.apply(self, dim)

    def squeeze(self, dim: int = 0) -> Tensor:
        return Squeeze.apply(self, dim)

    def transpose(self, n: int = -2, m: int = -1) -> Tensor:
        return Transpose.apply(self, n, m)

    def expand(self, shape: list[int]) -> Tensor:
        return Expand.apply(self, shape)

    def broadcast(self, shape: list[int]) -> "Tensor":
        if self.ndim > len(shape):
            raise ValueError(
                f"broadcast() error: source tensor has higher rank than target shape."
            )

        t = self  # Start with self

        # Add dimensions at the beginning
        for i in range(len(shape) - self.ndim):
            t = t.unsqueeze(0)

        # Check if broadcasting is possible
        for i in range(len(shape)):
            if shape[i] != t.shape[i] and t.shape[i] != 1:
                raise RuntimeError(
                    f"broadcast() error: can't broadcast {self.shape} to {shape}"
                )

        z = t.expand(shape)
        return z

    def relu(self) -> Tensor:
        return ReLU.apply(self)

    def log(self) -> Tensor:
        return Log.apply(self)

    def exp(self) -> Tensor:
        return Exp.apply(self)

    def softmax(self) -> Tensor:
        return Softmax.apply(self)

    def abs(self) -> Tensor:
        return Abs.apply(self)

    def realize(self):
        graph = self._node.topo_sort()
        self._node.realize(graph)

    def backward(self):
        c_set_ones_grad(self._c_tensor)
        graph = self._node.topo_sort()

        self._node.realize(graph)

        self._node.backward(graph[::-1])

    @staticmethod
    def safe_c_numel(shape_ptr, ndim):
        if ndim == 0:
            return 1

        if ndim < 0 or not shape_ptr:
            return 0

        try:
            size = 1
            for i in range(ndim):
                dim_size = shape_ptr[i]
                if dim_size <= 0:
                    return 0
                size *= dim_size
            return size
        except (IndexError, ValueError, ctypes.ArgumentError):
            return 0

    def numel(self):
        return Tensor.safe_c_numel(self._c_tensor.contents.shape, self.ndim)

    def __del__(self):
        if self._c_tensor:
            c_free_tensor(self._c_tensor)


if __name__ == "__main__":
    x = Tensor((2, 2), [[1, 2], [3, 4]])
    m = Tensor((2, 2), [[1, 2], [3, 4]])

    j = x + m

    j.backward()

    print(j)
    print(j.grad)

    y = Tensor((1, 4), [[3], [4], [1], [2]])
    n = x.view([1, 4])
    z = n + y

    z.realize()

    print(z)
