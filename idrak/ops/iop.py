from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import ctypes
import numpy as np

from idrak.ops.op import LazyOp
from idrak.core.tensor import Tensor
from idrak.core.buffer import LazyBuffer
from idrak.idrak_bindings.ctypes_definitions import CTensor
from idrak.idrak_bindings.c_wrapper_functions import (
    c_ones, c_zeros, c_uniform, c_randn, c_from_data
)

class IOp(LazyOp):
    @classmethod
    def calc_out_shape(cls, *args, **kwargs) -> Tuple[int, ...]:
        if not args:
            raise ValueError("IOp.calc_out_shape requires at least a shape argument.")
        
        shape = args[0]
        if not isinstance(shape, (tuple, list)):
            raise TypeError(f"Expected shape to be a tuple or list, but got {type(shape)}")
        return tuple(shape)

    @classmethod
    def create_ctx_struct(cls, *args, **kwargs) -> Tuple[Dict[str, Any], Any]:
        forward_kwargs = {k: v for k, v in kwargs.items()}
        return forward_kwargs, None

    def forward(self, out: "Tensor", *args, **kwargs):
        raise NotImplementedError("IOp.forward must be implemented by subclasses.")

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any = None):
        pass


class Ones(IOp):
    @classmethod
    def create_ctx_struct(cls, *args, **kwargs) -> Tuple[Dict[str, Any], Any]:
        return {}, None

    def forward(self, out: "Tensor", *args, **kwargs):
        if not out.c_tensor_ptr.contents.data:
            c_ones(out.c_tensor_ptr)

class Zeros(IOp):
    @classmethod
    def create_ctx_struct(cls, *args, **kwargs) -> Tuple[Dict[str, Any], Any]:
        return {}, None

    def forward(self, out: "Tensor", *args, **kwargs):
        if not out.c_tensor_ptr.contents.data:
            c_zeros(out.c_tensor_ptr)

class Uniform(IOp):
    @classmethod
    def create_ctx_struct(cls, *args, **kwargs) -> Tuple[Dict[str, Any], Any]:
        low_val = kwargs.get('low', 0.0)
        high_val = kwargs.get('high', 1.0)
        forward_kwargs = {"low": low_val, "high": high_val}
        return forward_kwargs, None

    def forward(self, out: "Tensor", *args, **kwargs):
        low = kwargs.get("low", 0.0)
        high = kwargs.get("high", 1.0)
        if not out.c_tensor_ptr.contents.data:
            c_uniform(out.c_tensor_ptr, low, high)

class Randn(IOp):
    @classmethod
    def create_ctx_struct(cls, *args, **kwargs) -> Tuple[Dict[str, Any], Any]:
        return {}, None

    def forward(self, out: "Tensor", *args, **kwargs):
        if not out.c_tensor_ptr.contents.data:
            c_randn(out.c_tensor_ptr)

class FromData(IOp):
    @classmethod
    def create_ctx_struct(cls, *args, **kwargs) -> Tuple[Dict[str, Any], Any]:
        if 'data' not in kwargs:
            raise ValueError("FromData operation requires 'data' keyword argument.")
        
        forward_kwargs = {"data": kwargs['data']}
        return forward_kwargs, None

    def forward(self, out: "Tensor", *args, **kwargs):
        data_input = kwargs["data"]

        if isinstance(data_input, np.ndarray):
            c_array_type = ctypes.c_float * data_input.size
            data_ptr = data_input.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            num_elements = data_input.size
        elif isinstance(data_input, (list, tuple)):
            flat_data = []
            for item in data_input:
                if isinstance(item, (list, tuple)):
                    flat_data.extend(item)
                else:
                    flat_data.append(item)
            
            c_array_type = ctypes.c_float * len(flat_data)
            data_ptr = c_array_type(*flat_data)
            num_elements = len(flat_data)
        else:
            raise TypeError(f"Unsupported data type for FromData: {type(data_input)}. Expected list, tuple, or numpy.ndarray.")

        if not out.c_tensor_ptr.contents.data:
            c_from_data(out.c_tensor_ptr, data_ptr)

