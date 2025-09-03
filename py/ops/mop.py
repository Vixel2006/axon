from __future__ import annotations
from os import wait
from typing import Any
import ctypes
from .op import LazyOp
from py.elnawah_bindings.ctypes_definitions import CTensor
from py.elnawah_bindings.c_wrapper_functions import (
    c_view,
    c_unsqueeze,
    c_squeeze,
    c_expand,
    c_broadcast,
    c_transpose,
    c_concat,
    c_stack
)

class View(LazyOp):
    @staticmethod
    def calc_out_shape(a: "Tensor", shape: tuple[int, ...]) -> tuple[int, ...]:
        from py.core.tensor import Tensor

        if a.numel() != Tensor.safe_c_numel(shape, len(shape)):
            raise RuntimeError(
                f"Unable to operate view as {a.numel()} != {Tensor.safe_c_numel(shape, len(shape))}"
            )
        return tuple(shape)

    
    @staticmethod
    def forward(out: "Tensor", a: "Tensor", shape: tuple[int, ...]) -> "Tensor":
        c_view(a._c_tensor, out._c_tensor, shape, len(shape))
        return out

class Unsqueeze(LazyOp):
    @staticmethod
    def calc_out_shape(a: "Tensor", dim: int) -> tuple[int, ...]:
        if dim < 0:
            dim = a.ndim + dim + 1
        if dim > a.ndim:
            raise ValueError(f"Can't unsqueeze dim {dim}.")
        new_shape = list(a.shape)
        new_shape.insert(dim, 1)
        return tuple(new_shape)

    @staticmethod
    def forward(out: "Tensor", a: "Tensor", dim: int) -> "Tensor": 
        c_unsqueeze(a._c_tensor, out._c_tensor, dim)
        return out


class Squeeze(LazyOp):
    @staticmethod
    def forward(out: "Tensor", a: "Tensor", dim: int) -> "Tensor": 
        c_squeeze(a._c_tensor, out._c_tensor, dim)
        return out


class Transpose(LazyOp):
    @staticmethod
    def forward(out: "Tensor", a: "Tensor", n: int, m: int) -> "Tensor":
        c_transpose(a._c_tensor, out._c_tensor, n, m)
        return out

class Expand(LazyOp):
    @staticmethod
    def forward(out: "Tensor", a: "Tensor", shape: tuple[int, ...]) -> "Tensor":
        c_expand(a._c_tensor, out._c_tensor, shape)
        return out


class Broadcast(LazyOp):
    @staticmethod
    def forward(out: "Tensor", a: "Tensor", shape: tuple[int, ...], ndim: int) -> "Tensor":
        c_broadcast(a._c_tensor, out._c_tensor, ndim, shape)
        return out


class Concat(LazyOp):
    @staticmethod
    def calc_out_shape(a: list["Tensor"], axis: int) -> tuple[int, ...]:
        from py.core.tensor import Tensor

        shape = list(a[0].shape)



        for i in range(1, len(a)):
            shape[axis] += a[i].shape[axis]
            for j in range(len(a[0].shape)):
                if j != axis and a[i].shape[j] != shape[j]:
                    raise ValueError("Can't concat")
            


        return tuple(shape)

    
    @staticmethod
    def forward(out: "Tensor", a: list["Tensor"], axis: int) -> "Tensor":
        # Ensure that `t._c_tensor` are valid and managed correctly.
        # The C function expects valid pointers to CTensor objects.
        inputs = []
        for t in a:
            inputs.append(t._c_tensor)
        c_concat(inputs, out._c_tensor, len(inputs), axis)
        return out
