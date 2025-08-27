from __future__ import annotations

from .elnawah_bindings import (
    c_malloc_tensor_empty,
    c_malloc_tensor_shape,
    c_malloc_tensor_full,
    c_free_tensor,
    c_numel,
    c_compute_strides,
    c_relu,
    c_log,
    c_exp,
    c_softmax,
    c_abs,
    c_neg,
    c_add,
    c_sub,
    c_mul,
    c_div,
    c_add_scalar,
    c_sub_scalar,
    c_rsub_scalar,
    c_mul_scalar,
    c_div_scalar,
    c_rdiv_scalar,
    c_matmul,
    c_sum,
    c_mean,
    c_max,
    c_view,
    c_unsqueeze,
    c_squeeze,
    c_transpose,
    c_expand,
    c_add_grad_op,
    c_sub_grad_op,
    c_rsub_grad_op,
    c_mul_grad_op,
    c_div_grad_op,
    c_rdiv_grad_op,
    c_relu_grad_op,
    c_log_grad_op,
    c_exp_grad_op,
    c_abs_grad_op,
    c_neg_grad_op,
    CTensor,
    CNode,
    c_malloc_node,
    tensor_lib,
    BackwardFnType,  # Import BackwardFnType
)

from .node import Node

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


class Tensor:
    def __init__(self, shape=None, data=None, requires_grad=True, _c_tensor_ptr=None):
        if _c_tensor_ptr is not None:
            self._c_tensor = _c_tensor_ptr
            if self._c_tensor and self._c_tensor.contents:
                self._shape = [
                    self._c_tensor.contents.shape[i]
                    for i in range(self._c_tensor.contents.ndim)
                ]
            else:
                self._shape = None
            return

        self._c_tensor = None
        self._node = None

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
            n = int(np.prod(self._shape)) if self._shape else 0

            if n <= 0:
                raise ValueError(f"Invalid tensor size: {n}")

            flat_data = []
            for i in range(n):
                try:
                    flat_data.append(t.data[i])
                except (IndexError, ValueError, ctypes.ArgumentError) as e:
                    raise ValueError(f"Failed to access tensor data at index {i}: {e}")

            np_array = np.array(flat_data, dtype=np.float32)
            return np_array.reshape(self._shape)

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
        input_c_tensor = self._c_tensor.contents

        out_tensor = Tensor(shape=self.shape, requires_grad=self.requires_grad)
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise RuntimeError(
                    f"Can't add tensors with shapes {self.shape} and {other.shape}"
                )

            c_add(self._c_tensor, other._c_tensor, out_tensor._c_tensor)

            if out_tensor.requires_grad:
                out_tensor._node = Node(
                    out_tensor, [self, other], BackwardFnType(tensor_lib.add_grad_op)
                )
        else:
            c_add_scalar(self._c_tensor, other, out_tensor._c_tensor)

            if out_tensor.requires_grad:
                scalar_val = ctypes.c_float(other)
                out_tensor._node = Node(
                    out_tensor,
                    [self],
                    BackwardFnType(tensor_lib.add_grad_op),
                    ctypes.byref(scalar_val),
                )

        return out_tensor

    def __iadd__(self, other: Union[Tensor, float]) -> Tensor:
        return self + other

    def __radd__(self, other: float) -> Tensor:
        return self + other

    def __sub__(self, other: Union[Tensor, float]) -> Tensor:
        input_c_tensor = self._c_tensor.contents

        out_tensor = Tensor(shape=self.shape, requires_grad=self.requires_grad)

        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise RuntimeError(
                    f"Can't add tensors with shapes {self.shape} and {other.shape}"
                )

            c_sub(self._c_tensor, other._c_tensor, out_tensor._c_tensor)

            if out_tensor.requires_grad:
                out_tensor._node = Node(
                    out_tensor, [self, other], BackwardFnType(tensor_lib.sub_grad_op)
                )
        else:
            c_sub_scalar(self._c_tensor, other, out_tensor._c_tensor)

            if out_tensor.requires_grad:
                scalar_val = ctypes.c_float(other)
                out_tensor._node = Node(
                    out_tensor,
                    [self],
                    BackwardFnType(tensor_lib.sub_grad_op),
                    ctypes.byref(scalar_val),
                )

        return out_tensor

    def __isub__(self, other: Union[Tensor, float]) -> Tensor:
        return self - other

    def __rsub__(self, other: float) -> Tensor:
        out_tensor = Tensor(shape=self.shape, requires_grad=self.requires_grad)

        c_rsub_scalar(other, self._c_tensor, out_tensor._c_tensor)

        if out_tensor.requires_grad:
            scalar_val = ctypes.c_float(other)
            out_tensor._node = Node(
                out_tensor,
                [self],
                BackwardFnType(tensor_lib.rsub_grad_op),
                ctypes.byref(scalar_val),
            )

        return out_tensor

    def __mul__(self, other: Union[Tensor, float]) -> Tensor:
        input_c_tensor = self._c_tensor.contents

        out_tensor = Tensor(shape=self.shape, requires_grad=self.requires_grad)
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise RuntimeError(
                    f"Can't add tensors with shapes {self.shape} and {other.shape}"
                )

            c_mul(self._c_tensor, other._c_tensor, out_tensor._c_tensor)

            if out_tensor.requires_grad:
                out_tensor._node = Node(
                    out_tensor, [self, other], BackwardFnType(tensor_lib.mul_grad_op)
                )
        else:
            c_mul_scalar(self._c_tensor, other, out_tensor._c_tensor)

            if out_tensor.requires_grad:
                scalar_val = ctypes.c_float(other)
                out_tensor._node = Node(
                    out_tensor,
                    [self],
                    BackwardFnType(tensor_lib.mul_grad_op),
                    ctypes.byref(scalar_val),
                )

        return out_tensor

    def __imul__(self, other: Union[Tensor, float]) -> Tensor:
        return self * other

    def __rmul__(self, other: float) -> Tensor:
        return self * other

    def __truediv__(self, other: Union[Tensor, float]) -> Tensor:
        input_c_tensor = self._c_tensor.contents

        out_tensor = Tensor(shape=self.shape, requires_grad=self.requires_grad)
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise RuntimeError(
                    f"Can't add tensors with shapes {self.shape} and {other.shape}"
                )

            c_div(self._c_tensor, other._c_tensor, out_tensor._c_tensor)

            if out_tensor.requires_grad:
                out_tensor._node = Node(
                    out_tensor, [self, other], BackwardFnType(tensor_lib.div_grad_op)
                )
        else:
            c_div_scalar(self._c_tensor, other, out_tensor._c_tensor)

            if out_tensor.requires_grad:
                scalar_val = ctypes.c_float(other)
                out_tensor._node = Node(
                    out_tensor,
                    [self],
                    BackwardFnType(tensor_lib.div_grad_op),
                    ctypes.byref(scalar_val),
                )

        return out_tensor

    def __itruediv__(self, other: Union[Tensor, float]) -> Tensor:
        return self / other

    def __rtruediv__(self, other: float) -> Tensor:
        out_tensor = Tensor(shape=self.shape, requires_grad=self.requires_grad)
        c_rdiv_scalar(other, self._c_tensor, out_tensor._c_tensor)

        if out_tensor.requires_grad:
            scalar_val = ctypes.c_float(other)
            out_tensor._node = Node(
                out_tensor,
                [self],
                BackwardFnType(tensor_lib.rdiv_grad_op),
                ctypes.byref(scalar_val),
            )
        return out_tensor

    def __matmul__(self, other: Tensor) -> Tensor:
        input_c_tensor = self._c_tensor.contents

        N = self.shape[-2]
        M = self.shape[-1]

        if self.shape[-1] != other.shape[-2]:
            raise RuntimeError(
                f"Can't add tensors with shapes {self.shape} and {other.shape}"
            )

        K = self.shape[-1]
        P = other.shape[-1]

        out_c_tensor_ptr = c_malloc_tensor_empty()

        c_matmul(self._c_tensor, other._c_tensor, out_c_tensor_ptr, N, K, P)

        if not out_c_tensor_ptr:
            raise RuntimeError("Failed to allocate empty C tensor for view operation.")

        return Tensor(_c_tensor_ptr=out_c_tensor_ptr)

    def __neg__(self) -> Tensor:
        out_tensor = Tensor(shape=self.shape, requires_grad=self.requires_grad)
        c_neg(self._c_tensor, out_tensor._c_tensor)
        if out_tensor.requires_grad:
            out_tensor._node = Node(
                out_tensor, [self], BackwardFnType(tensor_lib.neg_grad_op)
            )
        return out_tensor

    def sum(self, axis: int = 0, keepdim: bool = True):
        out_c_tensor_ptr = c_malloc_tensor_empty()
        if not out_c_tensor_ptr:
            raise RuntimeError("Failed to allocate empty C tensor for view operation.")

        c_sum(self._c_tensor, out_c_tensor_ptr, axis, keepdim)

        return Tensor(_c_tensor_ptr=out_c_tensor_ptr)

    def mean(self, axis: int = 0, keepdim: bool = True):
        out_c_tensor_ptr = c_malloc_tensor_empty()
        if not out_c_tensor_ptr:
            raise RuntimeError("Failed to allocate empty C tensor for view operation.")

        c_mean(self._c_tensor, out_c_tensor_ptr, axis, keepdim)

        return Tensor(_c_tensor_ptr=out_c_tensor_ptr)

    def max(self, axis: int = 0, keepdim: bool = True):
        out_c_tensor_ptr = c_malloc_tensor_empty()
        if not out_c_tensor_ptr:
            raise RuntimeError("Failed to allocate empty C tensor for view operation.")

        c_max(self._c_tensor, out_c_tensor_ptr, axis, keepdim)

        return Tensor(_c_tensor_ptr=out_c_tensor_ptr)

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
        input_c_tensor = self._c_tensor.contents
        if c_numel(shape, len(shape)) != c_numel(self.shape, self.ndim):
            raise RuntimeError(
                f"Unable to operate view as {c_numel(shape, len(shape))} != {c_numel(self.shape, self.ndim)}"
            )

        out_c_tensor_ptr = c_malloc_tensor_empty()
        if not out_c_tensor_ptr:
            raise RuntimeError("Failed to allocate empty C tensor for view operation.")

        c_view(self._c_tensor, out_c_tensor_ptr, shape, len(shape))

        return Tensor(_c_tensor_ptr=out_c_tensor_ptr)

    def unsqueeze(self, dim: int = 0) -> Tensor:
        input_c_tensor = self._c_tensor.contents

        if dim < 0:
            dim = self.ndim + dim + 1

        if dim > self.ndim:
            raise ValueError(f"Can't unsqueeze dim {dim}.")

        out_c_tensor_ptr = c_malloc_tensor_empty()
        if not out_c_tensor_ptr:
            raise RuntimeError("Failed to allocate empty C tensor for view operation.")

        c_unsqueeze(self._c_tensor, out_c_tensor_ptr, dim)

        return Tensor(_c_tensor_ptr=out_c_tensor_ptr)

    def squeeze(self, dim: int = 0) -> Tensor:
        input_c_tensor = self._c_tensor.contents

        if dim < 0:
            dim = self.ndim + dim

        if dim >= self.ndim:
            raise ValueError(f"Can't squeeze dim {dim} as it doesn't exists.")

        if self.shape[dim] != 1:
            raise RuntimeError(
                f"Tensors can be squeezed only on shape[dim] = 1, dim = {self.shape[dim]}"
            )

        out_c_tensor_ptr = c_malloc_tensor_empty()
        if not out_c_tensor_ptr:
            raise RuntimeError("Failed to allocate empty C tensor for view operation.")

        c_squeeze(self._c_tensor, out_c_tensor_ptr, dim)

        return Tensor(_c_tensor_ptr=out_c_tensor_ptr)

    def transpose(self, n: int = -2, m: int = -1) -> Tensor:
        input_c_tensor = self._c_tensor.contents

        if n < 0:
            n = self.ndim + n
        if m < 0:
            m = self.ndim + m

        if n >= self.ndim or m >= self.ndim:
            raise ValueError(f"Can't transpose around non-exsiting axes {n},{m}")

        out_c_tensor_ptr = c_malloc_tensor_empty()
        if not out_c_tensor_ptr:
            raise RuntimeError("Failed to allocate empty C tensor for view operation.")

        c_transpose(self._c_tensor, out_c_tensor_ptr, n, m)

        return Tensor(_c_tensor_ptr=out_c_tensor_ptr)

    def expand(self, shape: list[int]) -> Tensor:
        input_c_tensor = self._c_tensor.contents

        if self.ndim != len(shape):
            raise ValueError(f"expand() error: Dimensionality mismatch")

        for i in range(self.ndim):
            if self.shape[i] != shape[i] and self.shape[i] != 1:
                raise RuntimeError(
                    f"expand() error: Can't expand dim {i} from {self.shape[i]} to {shape[i]}, Only dims of 1 can be expanded."
                )

        out_c_tensor_ptr = c_malloc_tensor_empty()
        if not out_c_tensor_ptr:
            raise RuntimeError("Failed to allocate empty C tensor for view operation.")

        c_expand(self._c_tensor, out_c_tensor_ptr, shape)

        return Tensor(_c_tensor_ptr=out_c_tensor_ptr)

    def broadcast(self, shape: list[int]) -> Tensor:
        input_c_tensor = self._c_tensor.contents

        if self.ndim > len(shape):
            raise ValueError(
                f"broadcast() error: source tensor has higher rank than target shape."
            )

        for i in range(len(shape) - self.ndim):
            t = self.unsqueeze(0)

        for i in range(len(shape)):
            if shape[i] != t.shape[i] and t.shape[i] != 1:
                raise RuntimeError(
                    f"broadcast() error: can't broadcast {self.shape} to {shape}"
                )

        z = t.expand(shape)

        return z

    def relu(self) -> Tensor:
        out_tensor = Tensor(shape=self.shape, requires_grad=self.requires_grad)
        c_relu(self._c_tensor, out_tensor._c_tensor)
        if out_tensor.requires_grad:
            out_tensor._node = Node(
                out_tensor, [self], BackwardFnType(tensor_lib.relu_grad_op)
            )
        return out_tensor

    def log(self) -> Tensor:
        out_tensor = Tensor(shape=self.shape, requires_grad=self.requires_grad)
        c_log(self._c_tensor, out_tensor._c_tensor)
        if out_tensor.requires_grad:
            out_tensor._node = Node(
                out_tensor, [self], BackwardFnType(tensor_lib.log_grad_op)
            )
        return out_tensor

    def exp(self) -> Tensor:
        out_tensor = Tensor(shape=self.shape, requires_grad=self.requires_grad)
        c_exp(self._c_tensor, out_tensor._c_tensor)
        if out_tensor.requires_grad:
            out_tensor._node = Node(
                out_tensor, [self], BackwardFnType(tensor_lib.exp_grad_op)
            )
        return out_tensor

    def softmax(self) -> Tensor:
        out_tensor = Tensor(shape=self.shape, requires_grad=self.requires_grad)
        c_softmax(self._c_tensor, out_tensor._c_tensor)
        if out_tensor.requires_grad:
            out_tensor._node = Node(
                out_tensor, [self], BackwardFnType(tensor_lib.softmax_grad_op)
            )
        return out_tensor

    def abs(self) -> Tensor:
        out_tensor = Tensor(shape=self.shape, requires_grad=self.requires_grad)
        c_abs(self._c_tensor, out_tensor._c_tensor)
        if out_tensor.requires_grad:
            out_tensor._node = Node(
                out_tensor, [self], BackwardFnType(tensor_lib.abs_grad_op)
            )
        return out_tensor


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


if __name__ == "__main__":
    t = Tensor([2, 2, 3], [[[2, 3, 4], [3, 4, 5]], [[2, 3, 4], [3, 4, 5]]])

    n = t.log()

    n._node.backward()

    print(n.grad)
    print("---------------------")
    print(t.grad)
