from .elnawah_bindings import (
    c_malloc_tensor_empty,
    c_malloc_tensor_shape,
    c_malloc_tensor_full,
    c_free_tensor,
    c_numel,
    c_compute_strides,
    CTensor,
)

import numpy as np
import ctypes


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
    def __init__(self, shape=None, data=None, requires_grad=True):
        self._c_tensor = None

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
    n = Tensor([2, 3])
    print(n.shape)
    print(n.data)
    print(n.grad)
    t = Tensor([2, 3], [[2, 3, 4], [3, 4, 5]])

    print(t.shape)
    print(t.data)
    print(t.grad)
