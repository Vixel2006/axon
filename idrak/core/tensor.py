from __future__ import annotations

from idrak.idrak_bindings.c_wrapper_functions import (
    c_tmalloc,
    c_tfree,
    c_gmalloc,
    c_numel,
    c_compute_strides,
)
from idrak.idrak_bindings.ctypes_definitions import CTensor, CDevice
from idrak.idrak_bindings.c_library_loader import tensor_lib

from idrak.ops.uop import *
from idrak.ops.bop import *
from idrak.ops.mop import *
from idrak.ops.rop import *

import numpy as np
import ctypes
from typing import List, Optional, Tuple, Union
from enum import Enum

class Tensor:
    _lazy_buffer: Optional[Any]
    
    def __init__(self, shape: Tuple[int], device: str = "cpu", requires_grad: bool = True):
        device_ = 0 if device == "cpu" else 1
        ndim = len(shape)
        self.c_tensor_ptr = c_tmalloc(shape, ndim, device_, requires_grad)
        if not self.c_tensor_ptr:
            raise RuntimeError("tmalloc failed to allocate tensor")

        self._lazy_buffer = None

        if requires_grad:
            c_gmalloc(self.c_tensor_ptr, ctypes.c_float(0.0))

    @property
    def data(self) -> np.ndarray:
        out_array = np.empty(self.shape, dtype=np.float32)

        c_raw_data_ptr = self.c_tensor_ptr.contents.data.contents.data
        c_strides_ptr = self.c_tensor_ptr.contents.strides

        it = np.nditer(out_array, flags=['multi_index'], op_flags=['writeonly'])
        while not it.finished:
            logical_coords = it.multi_index
            
            c_memory_offset_elements = 0
            for i, coord in enumerate(logical_coords):
                c_memory_offset_elements += coord * c_strides_ptr[i]

            value = c_raw_data_ptr[c_memory_offset_elements]
            
            out_array[logical_coords] = value
            it.iternext()

        return out_array.copy()

    @property
    def grad(self) -> np.ndarray:
        out_array = np.empty(self.shape, dtype=np.float32)

        c_raw_data_ptr = self.c_tensor_ptr.contents.grad.contents.data
        c_strides_ptr = self.c_tensor_ptr.contents.strides

        it = np.nditer(out_array, flags=['multi_index'], op_flags=['writeonly'])
        while not it.finished:
            logical_coords = it.multi_index
            
            c_memory_offset_elements = 0
            for i, coord in enumerate(logical_coords):
                c_memory_offset_elements += coord * c_strides_ptr[i]

            value = c_raw_data_ptr[c_memory_offset_elements]
            
            out_array[logical_coords] = value
            it.iternext()

        return out_array.copy()

    @property
    def shape(self) -> Tuple[int]:
        return tuple(self.c_tensor_ptr.contents.shape[i] for i in range(self.c_tensor_ptr.contents.ndim))

    @property
    def strides(self) -> Tuple[int]:
        return tuple(self.c_tensor_ptr.contents.shape[i] for i in range(self.c_tensor_ptr.contents.ndim))

    @property
    def ndim(self) -> int:
        return self.c_tensor_ptr.contents.ndim

    @property
    def device(self) -> str:
        if self.c_tensor_ptr.contents.device == 0:
            return "cpu"
        else:
            return "cuda"

    @property
    def requires_grad(self) -> bool:
        return self.c_tensor_ptr.contents.requires_grad

    @property
    def T(self) -> Tensor:
        return Transpose.create_node(self, -2, -1)

    def realize(self) -> Tensor:
        if self._lazy_buffer is not None:
            return self._lazy_buffer.realize()
        return self

    def detach(self):
        self._lazy_buffer = None
        return self

    def backward(self):
        if self._lazy_buffer is not None:
            self._lazy_buffer.backward()

            execution_order = self._lazy_buffer.topo_sort()


    def numel(self) -> int: return c_numel(self.shape, self.ndim)

    def view(self, shape: tuple[int, ...]) -> Tensor: return View.create_node(self, shape)
    def unsqueeze(self, dim: int = -1) -> Tensor: return Unsqueeze.create_node(self, dim)
    def squeeze(self, dim: int = -1) -> Tensor: return Squeeze.create_node(self, dim)
    def expand(self, shape: tuple[int, ...]) -> Tensor: return Expand.create_node(self, shape)
    def broadcast(self, shape: tuple[int, ...]) -> Tensor: return Broadcast.create_node(self, shape)
    def transpose(self, n: int = -2, m: int = -1) -> Tensor: return Transpose.create_node(self, n, m)
    
    def exp(self) -> Tensor: return Exp.create_node(self)
    def log(self) -> Tensor: return Log.create_node(self)
    def abs(self) -> Tensor: return Abs.create_node(self)
    def relu(self) -> Tensor: return ReLU.create_node(self)

    def __add__(self, other: Tensor | float) -> Tensor: return Add.create_node(self, other)
    def __sub__(self, other: Tensor | float) -> Tensor: return Sub.create_node(self, other)
    def __mul__(self, other: Tensor | float) -> Tensor: return Mul.create_node(self, other)
    def __truediv__(self, other: Tensor | float) -> Tensor: return Div.create_node(self, other)
    def __pow__(self, other: Tensor | float) -> Tensor: return Pow.create_node(self, other)
    def __matmul__(self, other: Tensor) -> Tensor: return MatMul.create_node(self, other)
    def __radd__(self, other: float) -> Tensor: return Add.create_node(other, self)
    def __rmul__(self, other: float) -> Tensor: return Mul.create_node(other, self)
    def __rsub__(self, other: float) -> Tensor: return RSub.create_node(other, self)
    def __rtruediv__(self, other: float) -> Tensor: return RDiv.create_node(other, self)
    def __neg__(self) -> Tensor: return Neg.create_node(self)

    def __str__(self) -> str:
        return f"Tensor(shape={self.shape}, data={self.data}, device={self.device}, requires_grad={self.requires_grad})"

    def __del__(self):
        self._lazy_buffer = None
        if self.c_tensor_ptr:
            c_tfree(self.c_tensor_ptr)
            self.c_tensor_ptr = None
