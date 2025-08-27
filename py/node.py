from __future__ import annotations

from .elnawah_bindings.c_wrapper_functions import c_malloc_node, c_free_node
from .elnawah_bindings.ctypes_definitions import CTensor, CNode, BackwardFnType

import ctypes


class Node:
    def __init__(
        self,
        out_tensor: "Tensor",
        input_tensors: list["Tensor"],
        backward_fn,
        extras=None,
    ):
        self.out_tensor = out_tensor
        self.input_tensors = input_tensors
        self.backward_fn = backward_fn
        self.extras = extras
        self._extras_obj = None

        c_out_tensor_ptr = out_tensor._c_tensor

        n_prev = len(input_tensors)
        c_prev_array = (ctypes.POINTER(CTensor) * n_prev)()
        for i, t in enumerate(input_tensors):
            c_prev_array[i] = t._c_tensor

        c_extras = None
        if isinstance(extras, ctypes._Pointer) or isinstance(extras, ctypes._CFuncPtr):
            self._extras_obj = extras
            c_extras = extras
        elif isinstance(extras, ctypes._SimpleCData):
            self._extras_obj = extras
            c_extras = ctypes.byref(extras)
        elif extras is not None:
            c_extras = extras

        self._c_node = c_malloc_node(
            c_out_tensor_ptr,
            c_prev_array,
            n_prev,
            c_extras,
            BackwardFnType(backward_fn),
        )

        if (
            self._c_node
            and self.out_tensor._c_tensor
            and self.out_tensor._c_tensor.contents
        ):
            self.out_tensor._c_tensor.contents.ctx = self._c_node

    def backward(self):
        c_prev_array = (ctypes.POINTER(CTensor) * len(self.input_tensors))()
        for i, t in enumerate(self.input_tensors):
            c_prev_array[i] = t._c_tensor

        if isinstance(self._extras_obj, ctypes._SimpleCData):
            extras_to_pass = ctypes.byref(self._extras_obj)
        else:
            extras_to_pass = self._extras_obj

        self.backward_fn(
            self.out_tensor._c_tensor,
            c_prev_array,
            len(self.input_tensors),
            extras_to_pass,
        )

