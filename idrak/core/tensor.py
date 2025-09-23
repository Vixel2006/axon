from __future__ import annotations

from idrak.idrak_bindings.c_wrapper_functions import (
    c_tmalloc,
    c_tfree,
    c_gmalloc,
    c_gfree,
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

    def broadcast(self, shape):
        return Broadcast.create_node(self, shape)

    def realize(self) -> Tensor:
        if self._lazy_buffer is not None:
            return self._lazy_buffer.realize()
        return self

    def numel(self):
        return c_numel(self.shape, self.ndim)

    def __str__(self) -> str:
        return f"Tensor(shape={self.shape}, data={self.data}, device={self.device}, requires_grad={self.requires_grad})"

    def __del__(self):
        if self.c_tensor_ptr:
            c_tfree(self.c_tensor_ptr)
            self.c_tensor_ptr = None

