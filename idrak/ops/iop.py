# In idrak/ops/creation_ops.py (or similar file)
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import ctypes
import numpy as np

from idrak.ops.op import LazyOp
from idrak.core.tensor import Tensor
from idrak.core.buffer import LazyBuffer
from idrak.idrak_bindings.ctypes_definitions import CTensor
from idrak.idrak_bindings.c_wrapper_functions import (
    c_ones, c_zeros, c_uniform, c_randn,
)


class IOp(LazyOp):

    @staticmethod
    def calc_out_shape(*args, **kwargs) -> Tuple[int, ...]:
        if not args:
            raise ValueError("IOp.calc_out_shape requires at least a shape arguments.")
        
        shape = args[0]
        if not isinstance(shape, (tuple, list)):
            raise TypeError(f"Expected shape to be a tuple or list, but got {type(shape)}")
        return tuple(shape)

    @staticmethod
    def create_ctx_struct(*args, **kwargs) -> Tuple[Dict[str, Any], Any]:
        forward_kwargs = {k: v for k, v in kwargs.items()}
        return forward_kwargs, None

    @staticmethod
    def forward(out: "Tensor", **kwargs):
        raise NotImplementedError("IOp.forward must be implemented by subclasses.")

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any = None):
        pass


class Ones(IOp):
    @staticmethod
    def create_ctx_struct(*args, **kwargs) -> Tuple[Dict[str, Any], Any]:
        return {}, None

    @staticmethod
    def forward(out: "Tensor"):
        c_ones(out.c_tensor_ptr)

class Zeros(IOp):
    @staticmethod
    def create_ctx_struct(*args, **kwargs) -> Tuple[Dict[str, Any], Any]:
        return {}, None

    @staticmethod
    def forward(out: "Tensor"):
        c_zeros(out.c_tensor_ptr)

class Uniform(IOp):
    @staticmethod
    def create_ctx_struct(*args, **kwargs) -> Tuple[Dict[str, Any], Any]:
        forward_kwargs = {"low": kwargs['low'], "high": kwargs['high']}
        return forward_kwargs, None

    @staticmethod
    def forward(out: "Tensor", *args, **kwargs):
        c_uniform(out.c_tensor_ptr, kwargs["low"], kwargs["high"])

class Randn(IOp):
    @staticmethod
    def create_ctx_struct(*args, **kwargs) -> Tuple[Dict[str, Any], Any]:
        return {}, None

    @staticmethod
    def forward(out: "Tensor"):
        c_randn(out.c_tensor_ptr)

