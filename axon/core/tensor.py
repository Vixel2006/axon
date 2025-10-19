from __future__ import annotations

from axon.axon_bindings import c_wrapper_functions
from axon.axon_bindings.ctypes_definitions import CTensor, CDevice
from axon.axon_bindings.c_library_loader import tensor_lib
from axon.core.device import Device, get_default_device

from axon.ops import uop
from axon.ops import bop
from axon.ops import mop
from axon.ops import rop

import numpy as np
import ctypes
from typing import List, Optional, Tuple, Any
from enum import Enum

class Tensor:
    _lazy_buffer: Optional[Any]
    _grad: Optional[Tensor] = None
    
    def __init__(self, shape: Tuple[int], device: Device | None = None, requires_grad: bool = True, c_tensor_ptr: ctypes.POINTER(CTensor) | None = None, is_grad_view: bool = False):
        if device is None:
            device = get_default_device()

        ndim = len(shape)
        if not c_tensor_ptr:
            self.c_tensor_ptr = c_wrapper_functions.c_tmalloc(shape, ndim, device, requires_grad)
        else:
            self.c_tensor_ptr = c_tensor_ptr

        if not self.c_tensor_ptr:
            raise RuntimeError("tmalloc failed to allocate tensor")

        self._lazy_buffer = None
        self._grad = None
        self.device = device
        self.is_grad_view = is_grad_view

    @property
    def data(self) -> np.ndarray:
        num_elements = self.c_tensor_ptr.contents.data.contents.size
        out_array = np.empty(self.shape, dtype=np.float32)

        if self.device.type == "cpu":
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
        else:
            host_buffer_ptr = out_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            c_wrapper_functions.c_copy_storage_to_host(
                self.c_tensor_ptr.contents.data,
                self.device,
                num_elements,
                host_buffer_ptr
            )

        return out_array

    @property
    def grad(self) -> Tensor:
        if not self.c_tensor_ptr.contents.grad:
            return None
        if self._grad is None:
            self._grad = Tensor(self.shape, device=self.device, c_tensor_ptr=self.c_tensor_ptr.contents.grad, is_grad_view=True)
        return self._grad

    @property
    def shape(self) -> Tuple[int]:
        return tuple(self.c_tensor_ptr.contents.shape[i] for i in range(self.c_tensor_ptr.contents.ndim))

    @property
    def strides(self) -> Tuple[int]:
        return tuple(self.c_tensor_ptr.contents.strides[i] for i in range(self.c_tensor_ptr.contents.ndim))

    @property
    def ndim(self) -> int:
        return self.c_tensor_ptr.contents.ndim

    @property
    def requires_grad(self) -> bool:
        return self.c_tensor_ptr.contents.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool):
        self.c_tensor_ptr.contents.requires_grad = value

    @property
    def T(self) -> Tensor:
        return mop.Transpose.create_node(self, -2, -1)

    def realize(self) -> Tensor:
        if self._lazy_buffer is not None:
            return self._lazy_buffer.realize()
        return self

    def detach(self) -> Tensor:
        from axon.functions import from_c_storage
        self.realize()
        if not self.c_tensor_ptr.contents.data:
            return Tensor(self.shape, device=self.device, requires_grad=False)
        
        detached_tensor = from_c_storage(self.shape, self.c_tensor_ptr.contents.data, device=self.device, requires_grad=False)
        detached_tensor.c_tensor_ptr.contents.requires_grad = False 
        return detached_tensor

    def backward(self):
        if not self.requires_grad:
            raise RuntimeError("Gradient storage not allocated: cannot call backward on a tensor that does not require grad.")

        self.realize()

        if self._lazy_buffer is not None:
            self._lazy_buffer.backward()


    def numel(self) -> int: return c_wrapper_functions.c_numel(self.shape, self.ndim)

    def view(self, shape: tuple[int, ...]) -> Tensor: return mop.View.create_node(self, shape)
    def unsqueeze(self, dim: int = -1) -> Tensor: return mop.Unsqueeze.create_node(self, dim)
    def squeeze(self, dim: int = -1) -> Tensor: return mop.Squeeze.create_node(self, dim)
    def expand(self, shape: tuple[int, ...]) -> Tensor: return mop.Expand.create_node(self, shape)
    def broadcast(self, shape: tuple[int, ...]) -> Tensor: return mop.Broadcast.create_node(self, shape)
    def transpose(self, n: int = -2, m: int = -1) -> Tensor: return mop.Transpose.create_node(self, n, m)
    def flatten(self) -> Tensor: return mop.View.create_node(self, (self.numel(),))
    
    def exp(self) -> Tensor: return uop.Exp.create_node(self)
    def log(self) -> Tensor: return uop.Log.create_node(self)
    def abs(self) -> Tensor: return uop.Abs.create_node(self)
    def relu(self) -> Tensor: return uop.ReLU.create_node(self)

    def __add__(self, other: Tensor | float) -> Tensor: return bop.Add.create_node(self, other)
    def __sub__(self, other: Tensor | float) -> Tensor: return bop.Sub.create_node(self, other)
    def __mul__(self, other: Tensor | float) -> Tensor: return bop.Mul.create_node(self, other)
    def __truediv__(self, other: Tensor | float) -> Tensor: return bop.Div.create_node(self, other)
    def __pow__(self, other: Tensor | float) -> Tensor: return bop.Pow.create_node(self, other)
    def __matmul__(self, other: Tensor) -> Tensor: return bop.MatMul.create_node(self, other)
    def __radd__(self, other: float) -> Tensor: return bop.Add.create_node(self, other)
    def __rmul__(self, other: float) -> Tensor: return bop.Mul.create_node(self, other)
    def __rsub__(self, other: float) -> Tensor: return bop.RSub.create_node(other, self)
    def __rtruediv__(self, other: float) -> Tensor: return bop.RDiv.create_node(other, self)
    def __neg__(self) -> Tensor: return uop.Neg.create_node(self)

    def __str__(self) -> str:
        return f"Tensor(shape={self.shape}, data={self.data}, device={self.device!r}, requires_grad={self.requires_grad})"
    
    def __del__(self):
        self._lazy_buffer = None
        if hasattr(self, 'is_grad_view') and self.is_grad_view:
            return

        if self.c_tensor_ptr:
            c_wrapper_functions.c_tfree(self.c_tensor_ptr)
            self.c_tensor_ptr = None


if __name__ == "__main__":
    from axon.functions import zeros, ones
    
    device = Device("cuda")

    a = zeros((2,2), device=device)
    print(a.c_tensor_ptr.contents.device.contents.type)
