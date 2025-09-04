from __future__ import annotations

from typing import Optional, Callable, Tuple, Dict, Any

from py.idrak_bindings.c_wrapper_functions import c_malloc_node, c_free_node
from py.idrak_bindings.ctypes_definitions import CTensor, CNode

import ctypes


class Node:
    def __init__(
        self,
        out_tensor: "Tensor",
        input_tensors: list["Tensor"],
        forward_fn: Callable,
        forward_args: Tuple,
        forward_kwargs: Dict,
        backward_fn: Any = None,
        extras=None,
    ):
        self.out_tensor = out_tensor
        self.input_tensors = input_tensors
        self.forward_fn = forward_fn
        self.forward_args = forward_args
        self.forward_kwargs = forward_kwargs
        self.backward_fn = backward_fn
        self.extras = extras
        self._extras_obj = None
        self._c_node = None

        self._python_backward_fn = None
        c_backward_fn_ptr = ctypes.c_void_p(None)

        if callable(backward_fn):
            self._python_backward_fn = backward_fn
        elif backward_fn is not None:
            c_backward_fn_ptr = ctypes.cast(backward_fn, ctypes.c_void_p)

        if not hasattr(out_tensor, "_c_tensor") or out_tensor._c_tensor is None:
            return

        c_out_tensor_ptr = out_tensor._c_tensor

        n_prev = len(input_tensors)
        c_prev_array = (ctypes.POINTER(CTensor) * n_prev)()
        for i, t in enumerate(input_tensors):
            if hasattr(t, "_c_tensor") and t._c_tensor is not None:
                c_prev_array[i] = t._c_tensor
            else:
                return

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
            ctypes.c_void_p(None),
            c_backward_fn_ptr,
        )

        if (
            self._c_node
            and self.out_tensor._c_tensor
            and self.out_tensor._c_tensor.contents
        ):
            self.out_tensor._c_tensor.contents.ctx = self._c_node
            self.out_tensor._node = self

    def topo_sort(self) -> List[Node]:
        graph = []
        visited = set()

        def visit(node):
            if node in visited:
                return
            visited.add(node)

            for input_tensor in node.input_tensors:
                if hasattr(input_tensor, "_node") and input_tensor._node:
                    visit(input_tensor._node)

            graph.append(node)

        visit(self)

        return graph

    def realize(self, graph):
        for node in graph:
            if node.forward_fn:
                result_tensor = node.forward_fn(
                    *node.forward_args, **node.forward_kwargs
                )

                node.out_tensor._c_tensor = result_tensor._c_tensor

                if node.out_tensor._c_tensor and node.out_tensor._c_tensor.contents:
                    if node.out_tensor._c_tensor.contents.ndim == 0:
                        node.out_tensor._shape = ()
                    else:
                        node.out_tensor._shape = tuple(
                            [
                                node.out_tensor._c_tensor.contents.shape[i]
                                for i in range(node.out_tensor._c_tensor.contents.ndim)
                            ]
                        )

                if node._c_node and not node._c_node.contents.out:
                    node._c_node.contents.out = node.out_tensor._c_tensor

    def backward(self, graph):
        if not graph:
            return

        visited = set()

        for node in reversed(graph):
            if node in visited:
                continue
            visited.add(node)

            out_t = node.out_tensor
            if getattr(out_t, "grad", None) is None:
                continue

            n_prev = len(node.input_tensors)
            if n_prev > 0:
                c_prev_array = (ctypes.POINTER(CTensor) * n_prev)()
                for j, t in enumerate(node.input_tensors):
                    c_prev_array[j] = getattr(t, "_c_tensor", None)
            else:
                c_prev_array = None

            if isinstance(node._extras_obj, ctypes._SimpleCData):
                extras_to_pass = ctypes.byref(node._extras_obj)
            else:
                extras_to_pass = node._extras_obj

            if node._python_backward_fn:
                node._python_backward_fn(
                    node.out_tensor._c_tensor,
                    c_prev_array,
                    n_prev,
                    extras_to_pass,
                )
            elif node.backward_fn:
                node.backward_fn(
                    node.out_tensor._c_tensor,
                    c_prev_array,
                    n_prev,
                    extras_to_pass,
                )



    def __del__(self):
        if self._c_node:
            c_free_node(self._c_node)
