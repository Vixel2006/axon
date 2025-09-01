from __future__ import annotations

from py.elnawah_bindings.c_wrapper_functions import (
    c_malloc_tensor_empty,
    c_malloc_tensor_shape,
    c_malloc_tensor_full,
    c_free_tensor,
    c_numel,
    c_compute_strides,
    c_set_ones_grad,
)
from py.elnawah_bindings.ctypes_definitions import CTensor
from py.elnawah_bindings.c_library_loader import tensor_lib

from .node import Node
from py.ops.uop import *
from py.ops.bop import *
from py.ops.mop import *
from py.ops.rop import *

import numpy as np
import ctypes
from typing import Union


def _flatten_list(nested_list):
    """
    Recursively flattens a nested list or tuple into a single flat list.

    Parameters
    ----------
    nested_list : list | tuple | any
        The object to flatten. Non-list/tuple inputs are wrapped in a list.

    Returns
    -------
    list
        A one-dimensional list containing all elements of `nested_list`.
    """

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
    """
    Core tensor class for the autodiff engine.
    Wraps a C-allocated tensor (via ctypes) and provides Pythonic operators,
    gradient tracking, and integration with lazy execution nodes.
    """

    def __init__(self, shape=None, data=None, requires_grad=True, _c_tensor_ptr=None):
        """
        Create a new Tensor, backed by a C-allocated CTensor.

        Parameters
        ----------
        shape : tuple[int] | list[int] | None
            Shape of the tensor. If None, defaults to empty allocation.
        data : list[float] | np.ndarray | None
            Initial data to fill the tensor. If provided, shape must also be provided.
        requires_grad : bool, default=True
            Whether this tensor should participate in autograd.
        _c_tensor_ptr : ctypes.POINTER(CTensor) | None
            Low-level pointer to an already-allocated C tensor.
            Used internally for wrapping existing tensors.

        Raises
        ------
        ValueError
            If arguments are invalid (e.g., shape provided without data mismatch).
        """

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
            c_data = _flatten_list(data.tolist() if isinstance(data, np.ndarray) else data)
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
            current_shape = (
                self._shape
                if self._shape is not None
                else [t.shape[i] for i in range(t.ndim)]
            )

            return self._construct_data_with_strides(t, current_shape)

        except Exception as e:
            raise ValueError(f"Failed to convert tensor data to numpy array: {e}")

    def _construct_data_with_strides(self, t, shape):
        if not t.strides:
            raise ValueError("Invalid tensor: NULL strides pointer")

        result = np.zeros(shape, dtype=np.float32)
        
        indices = np.ndindex(tuple(shape))
        
        for idx in indices:
            flat_index = 0
            for i, coord in enumerate(idx):
                flat_index += coord * t.strides[i]
            
            try:
                result[idx] = t.data[flat_index]
            except (IndexError, ValueError, ctypes.ArgumentError) as e:
                raise ValueError(f"Failed to access tensor data at flat index {flat_index}: {e}")
        
        return result
    
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
            current_shape = (
                self._shape
                if self._shape is not None
                else [t.shape[i] for i in range(t.ndim)]
            )

            return self._construct_grad_with_strides(t, current_shape)

        except Exception as e:
            raise ValueError(f"Failed to convert tensor gradient to numpy array: {e}")

    def _construct_grad_with_strides(self, t, shape):
        if not t.strides:
            raise ValueError("Invalid tensor: NULL strides pointer")

        result = np.zeros(shape, dtype=np.float32)
        
        indices = np.ndindex(tuple(shape))
        
        for idx in indices:
            flat_index = 0
            for i, coord in enumerate(idx):
                flat_index += coord * t.strides[i]
            
            try:
                result[idx] = t.grad[flat_index]
            except (IndexError, ValueError, ctypes.ArgumentError) as e:
                raise ValueError(f"Failed to access tensor gradient at flat index {flat_index}: {e}")
        
        return result

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

    # ============ Binary Operations =============
    def __add__(self, other: Tensor | float) -> Tensor: return Add.create_node(self, other)
    def __sub__(self, other: Tensor | float) -> Tensor: return Sub.create_node(self, other)
    def __mul__(self, other: Tensor | float) -> Tensor: return Mul.create_node(self, other)
    def __matmul__(self, other: Tensor) -> Tensor: return MatMul.create_node(self, other)
    def __truediv__(self, other: Tensor | float) -> Tensor: return Div.create_node(self, other)
    def __rsub__(self, other: float) -> Tensor: return RSub.create_node(self, other)
    def __rtruediv__(self, other: float) -> Tensor: return RDiv.create_node(self, other)

    # ============ Unary Operations ==============
    def __neg__(self) -> Tensor: return Neg.create_node(self)
    def relu(self) -> Tensor: return ReLU.create_node(self)
    def log(self) -> Tensor: return Log.create_node(self)
    def exp(self) -> Tensor: return Exp.create_node(self)
    def abs(self) -> Tensor: return Abs.create_node(self)

    # ============ Movement Operations ==============
    def view(self, shape: tuple[int, ...]) -> Tensor: return View.create_node(self, shape=shape)
    def unsqueeze(self, dim: int = 0) -> Tensor: return Unsqueeze.create_node(self, dim=dim)
    def squeeze(self, dim: int = 0) -> Tensor: return Squeeze.create_node(self, dim=dim)
    def expand(self, shape: tuple[int, ...]) -> Tensor: return Expand.create_node(self, shape=shape)
    def broadcast(self, shape: tuple[int, ...]) -> Tensor: return Broadcast.create_node(self, shape=shape, ndim=len(shape))
    def transpose(self, n: int, m: int) -> Tensor: return Transpose.create_node(self, n=n, m=m)


    # ============ Reduction Operations ==============
    def sum(self, dim: int | None = None, keepdim: bool = False) -> Tensor: return Sum.create_node(self, dim=dim, keepdim=keepdim)
    def mean(self, dim: int | None = None, keepdim: bool = False) -> Tensor: return Mean.create_node(self, dim=dim, keepdim=keepdim)
    def max(self, dim: int | None = None, keepdim: bool = False) -> Tensor: return Max.create_node(self, dim=dim, keepdim=keepdim)

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
