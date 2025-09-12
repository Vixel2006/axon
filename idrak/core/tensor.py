from __future__ import annotations

from idrak.idrak_bindings.c_wrapper_functions import (
    c_malloc_tensor_shape,
    c_malloc_tensor_full,
    c_free_tensor,
    c_numel,
    c_compute_strides,
)
from idrak.idrak_bindings.ctypes_definitions import CTensor
from idrak.idrak_bindings.c_library_loader import tensor_lib

from .node import Node
from idrak.ops.uop import *
from idrak.ops.bop import *
from idrak.ops.mop import *
from idrak.ops.rop import *

import numpy as np
import ctypes
from typing import Union
from enum import Enum

class dtype(Enum):
    FLOAT32 = 0
    FLOAT64 = 1
    INT32 = 2

class Device(Enum):
    CPU = 0
    CUDA = 1

def _flatten_list(nested_list):
    if isinstance(nested_list, np.ndarray):
        return nested_list.flatten().tolist()
    if not isinstance(nested_list, (list, tuple)):
        return [nested_list]
    flattened = []
    for item in nested_list:
        if isinstance(item, (list, tuple, np.ndarray)):
            flattened.extend(_flatten_list(item))
        else:
            flattened.append(item)
    return flattened


class Tensor(CTensor):
    def __init__(
        self,
        shape=None,
        data=None,
        requires_grad=True,
        dtype: dtype = dtype.FLOAT32,
        device: str = "cpu",
        _c_tensor_ptr=None,
    ):
        self._dtype = dtype
        self._c_tensor = None
        self._shape = None 
        self._c_data_shared_ptr = None
        self._c_grad_shared_ptr = None

        if device == "cpu":
            self._device = Device.CPU
        elif device == "cuda":
            self._device = Device.CUDA
        else:
            raise ValueError("Backend not supported")

        self._node = None # Initialize _node to None

        if _c_tensor_ptr is not None:
            self._c_tensor = _c_tensor_ptr
            ndim = self._c_tensor.contents.ndim
            if ndim == 0: # Handle scalar tensors
                self._shape = ()
            else:
                self._shape = tuple(self._c_tensor.contents.shape[i] for i in range(ndim))
            if self._c_tensor.contents.data:
                self._c_data_shared_ptr = self._c_tensor.contents.data.contents
            if self._c_tensor.contents.grad:
                self._c_grad_shared_ptr = self._c_tensor.contents.grad.contents
        elif shape is not None and data is None:
            ndim = len(shape)
            self._shape = shape
            self._c_tensor = c_malloc_tensor_shape(shape, ndim, dtype.value, self.device.value, requires_grad)
            if self._c_tensor.contents.data:
                self._c_data_shared_ptr = self._c_tensor.contents.data.contents
            if self._c_tensor.contents.grad:
                self._c_grad_shared_ptr = self._c_tensor.contents.grad.contents
        elif shape is not None and data is not None:
            ndim = len(shape)
            self._shape = shape
            strides = c_compute_strides(shape, ndim)
            _c_tensor_ptr_temp = c_malloc_tensor_full(shape, ndim, strides, dtype.value, self.device.value, _flatten_list(data), requires_grad, None)
            self._c_tensor = _c_tensor_ptr_temp
            if self._c_tensor.contents.data:
                self._c_data_shared_ptr = self._c_tensor.contents.data.contents
            if self._c_tensor.contents.grad:
                self._c_grad_shared_ptr = self._c_tensor.contents.grad.contents

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self) -> dtype:
        return self._dtype

    @property
    def device(self) -> Device:
        return self._device

        
    def _get_ctypes_pointer_type(self, numpy_dtype):
        if numpy_dtype == np.float32:
            return ctypes.POINTER(ctypes.c_float)
        elif numpy_dtype == np.float64:
            return ctypes.POINTER(ctypes.c_double)
        elif numpy_dtype == np.int32:
            return ctypes.POINTER(ctypes.c_int32)
        else:
            raise ValueError(f"Unsupported numpy_dtype: {numpy_dtype}")

    @property
    def data(self) -> np.ndarray:
        if not self._c_tensor or not self._c_tensor.contents:
            raise ValueError("Invalid tensor: NULL pointer")

        t = self._c_tensor.contents
        
        numpy_dtype = self._get_numpy_dtype(self.dtype)
        c_pointer_type = self._get_ctypes_pointer_type(numpy_dtype)

        if self._shape == (1,):
            if not t.data or not t.data.contents or not t.data.contents.elems:
                raise ValueError("Invalid tensor: NULL data pointer")
            return np.array([ctypes.cast(t.data.contents.elems, c_pointer_type)[0]], dtype=numpy_dtype)

        if t.ndim < 0:
            raise ValueError(f"Invalid tensor: negative ndim {t.ndim}")

        if not t.shape:
            raise ValueError(
                "Invalid tensor: NULL shape pointer for multi-dimensional tensor"
            )

        if not t.data or not t.data.contents or not t.data.contents.elems:
            raise ValueError("Invalid tensor: NULL data pointer")

        try:
            current_shape = (
                self._shape
                if self._shape is not None
                else tuple(t.shape[i] for i in range(t.ndim))
            )
            return self._construct_data_with_strides(t, current_shape, numpy_dtype, c_pointer_type)

        except Exception as e:
            raise ValueError(f"Failed to convert tensor data to numpy array: {e}")

    def _construct_data_with_strides(self, t, shape, numpy_dtype, c_pointer_type):
        if not t.strides:
            raise ValueError("Invalid tensor: NULL strides pointer")

        result = np.zeros(shape, dtype=numpy_dtype)
        
        indices = np.ndindex(tuple(shape))
        
        for idx in indices:
            flat_index = 0
            for i, coord in enumerate(idx):
                flat_index += coord * t.strides[i]
            
            try:
                result[idx] = ctypes.cast(t.data.contents.elems, c_pointer_type)[flat_index]
            except (IndexError, ValueError, ctypes.ArgumentError) as e:
                raise ValueError(f"Failed to access tensor data at flat index {flat_index}: {e}")
        
        return result
    
    @property
    def grad(self) -> np.ndarray:
        if not self._c_tensor or not self._c_tensor.contents:
            raise ValueError("Invalid tensor: NULL pointer")

        t = self._c_tensor.contents
        numpy_dtype = self._get_numpy_dtype(self.dtype)
        c_pointer_type = self._get_ctypes_pointer_type(numpy_dtype)

        if not t.requires_grad or not t.grad or not t.grad.contents or not t.grad.contents.elems:
            return None

        if t.ndim == 0:
            return np.array(ctypes.cast(t.grad.contents.elems, c_pointer_type)[0], dtype=numpy_dtype)

        try:
            current_shape = (
                self._shape
                if self._shape is not None
                else tuple(t.shape[i] for i in range(t.ndim))
            )

            return self._construct_grad_with_strides(t, current_shape, numpy_dtype, c_pointer_type)

        except Exception as e:
            raise ValueError(f"Failed to convert tensor gradient to numpy array: {e}")

    def _construct_grad_with_strides(self, t, shape, numpy_dtype, c_pointer_type):
        if not t.strides:
            raise ValueError("Invalid tensor: NULL strides pointer")

        result = np.zeros(shape, dtype=numpy_dtype)
        
        indices = np.ndindex(tuple(shape))
        
        for idx in indices:
            flat_index = 0
            for i, coord in enumerate(idx):
                flat_index += coord * t.strides[i]
            
            try:
                result[idx] = ctypes.cast(t.grad.contents.elems, c_pointer_type)[flat_index]
            except (IndexError, ValueError, ctypes.ArgumentError) as e:
                raise ValueError(f"Failed to access tensor gradient at flat index {flat_index}: {e}")
        
        return result

    def _get_numpy_dtype(self, dtype: dtype):
        if dtype == dtype.FLOAT32:
            return np.float32
        elif dtype == dtype.FLOAT64:
            return np.float64
        elif dtype == dtype.INT32:
            return np.int32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

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
            grad_info = f", requires_grad={self.requires_grad}, dtype={self.dtype.name}, device={self.device.name}"
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
    def __pow__(self, other: Tensor | float) -> Tensor: return Pow.create_node(self, other)
    def __matmul__(self, other: Tensor) -> Tensor: return MatMul.create_node(self, other)
    def __truediv__(self, other: Tensor | float) -> Tensor: return Div.create_node(self, other)
    def __rsub__(self, other: float) -> Tensor: return RSub.create_node(self, other)
    def __rtruediv__(self, other: float) -> Tensor: return RDiv.create_node(self, other)
    def __radd__(self, other: float) -> Tensor: return Add.create_node(self, other)
    def __rmul__(self, other: float) -> Tensor: return Mul.create_node(self, other)
    def dot(self, other: Tensor) -> Tensor: return Dot.create_node(self, other)

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
    def sum(self, dim: int | None = None, keepdim: bool = True) -> Tensor: return Sum.create_node(self, dim=dim, keepdim=keepdim)
    def mean(self, dim: int | None = None, keepdim: bool = True) -> Tensor: return Mean.create_node(self, dim=dim, keepdim=keepdim)
    def max(self, dim: int | None = None, keepdim: bool = True) -> Tensor: return Max.create_node(self, dim=dim, keepdim=keepdim)

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
        graph = self._node.topo_sort()

        self._node.realize(graph)

        self._node.backward(graph)

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

    def item(self):
        if self.ndim == 0:
            return self.data.item()
        elif self.shape == (1,):
            return self.data[0]
        else:
            raise ValueError("only one element tensors can be converted to Python scalars")

    
