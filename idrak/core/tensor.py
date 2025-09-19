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
    def __init__(self, shape: Tuple[int], device: str = "cpu", requires_grad: bool = True):
        device_ = CDevice.CPU if device == "cpu" else CDevice.CUDA
        ndim = len(shape)
        c_shape = (ctypes.c_int * ndim)(*shape)
        self.c_tensor_ptr = c_tmalloc(c_shape, ndim, device_, requires_grad)
        if not self.c_tensor_ptr:
            raise RuntimeError("tmalloc failed to allocate tensor")

    @property
    def shape(self) -> Tuple[int]:
        return tuple(self.c_tensor_ptr.shape)

    @property
    def ndim(self) -> int:
        return len(self.c_tensor_ptr.ndim)

    @property
    def requries_grad(self) -> bool:
        return self.c_tensor_ptr.requires_grad


    def __str__(self):
        return f"Tensor(shape={self.shape}, device={self.device}, requires_grad={self.requires_grad}"

if __name__ == "__main__":
    t = Tensor((2,2))

    print(t)

