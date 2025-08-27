from __future__ import annotations

from typing import Optional

from ..elnawah_bindings.c_wrapper_functions import c_malloc_node, c_free_node
from ..elnawah_bindings.ctypes_definitions import CTensor, CNode, BackwardFnType

import ctypes


class Node:
    def __init__(
        self,
        out_tensor: "Tensor",
        input_tensors: list["Tensor"],
        backward_fn: Optional[BackwardFnType] = None,
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
            BackwardFnType(backward_fn) if backward_fn else BackwardFnType(0),
        )

        if (
            self._c_node
            and self.out_tensor._c_tensor
            and self.out_tensor._c_tensor.contents
        ):
            self.out_tensor._c_tensor.contents.ctx = self._c_node
            self.out_tensor._node = self

    def topo_sort(self):
        topo = []
        visited = set()

        def visit(node):
            if node in visited:
                return
            visited.add(node)

            for input_tensor in node.input_tensors:
                if input_tensor._node:
                    visit(input_tensor._node)

            topo.append(node)

        visit(self)

        return topo

    def realize(self):
        pass

    def backward(self):
        c_prev_array = (ctypes.POINTER(CTensor) * len(self.input_tensors))()
        for i, t in enumerate(self.input_tensors):
            c_prev_array[i] = t._c_tensor

        if isinstance(self._extras_obj, ctypes._SimpleCData):
            extras_to_pass = ctypes.byref(self._extras_obj)
        else:
            extras_to_pass = self._extras_obj

        if self.backward_fn:
            self.backward_fn(
                self.out_tensor._c_tensor,
            c_prev_array,
            len(self.input_tensors),
            extras_to_pass,
        )

        # Recursively call backward on input nodes
        for input_tensor in self.input_tensors:
            if input_tensor.requires_grad and input_tensor._node:
                input_tensor._node.backward()


if __name__ == "__main__":
    from .tensor import Tensor

    def dummy_backward(*args):
        pass

    a = Tensor(shape=(1,), data=[1.0], requires_grad=True)
    b = Tensor(shape=(1,), data=[2.0], requires_grad=True)
    c = Tensor(shape=(1,), data=[3.0], requires_grad=True)
    d = Tensor(shape=(1,), data=[4.0], requires_grad=True)

    node_c = Node(out_tensor=c, input_tensors=[a, b], backward_fn=dummy_backward)
    c._node = node_c

    node_d = Node(out_tensor=d, input_tensors=[c], backward_fn=dummy_backward)
    d._node = node_d

    print("Performing topological sort starting from node_d:")
    sorted_nodes = node_d.topo_sort()

    print(
        "Topologically sorted nodes (should be c, a, b, d or similar order of a,b before c, and c before d):"
    )
    for node in sorted_nodes:
        print(f"Node producing tensor with data: {node.out_tensor.data}")

