from __future__ import annotations

from typing import Optional, Callable, Tuple, Dict, Any

from ..elnawah_bindings.c_wrapper_functions import c_malloc_node, c_free_node
from ..elnawah_bindings.ctypes_definitions import CTensor, CNode

import ctypes


class Node:
    """
    Node:
    Represents a computation node in the lazy execution graph.
    Each node tracks the output tensor, input tensors, forward function,
    backward function, and any extra data needed for execution.

    Parameters
    ----------
    out_tensor : Tensor
        The output tensor produced by this node.
    input_tensors : list[Tensor]
        List of input tensors consumed by this node.
    forward_fn : Callable
        The forward function to compute the output.
    forward_args : Tuple
        Positional arguments passed to the forward function.
    forward_kwargs : Dict
        Keyword arguments passed to the forward function.
    backward_fn : Any, optional
        Function or pointer to the backward function used for gradient computation.
        If callable, treated as a Python backward function; otherwise, assumed
        to be a C function pointer. Default is None.
    extras : Any, optional
        Additional context or data required for forward/backward execution.
        Can be ctypes objects, C function pointers, or Python objects.

    Attributes
    ----------
    out_tensor : Tensor
        Output tensor reference.
    input_tensors : list[Tensor]
        Input tensor references.
    forward_fn : Callable
        Forward function reference.
    backward_fn : Any
        Backward function reference (Python or C).
    extras : Any
        Extra context data.
    _extras_obj : Any
        Internal storage for extras if passed as ctypes data.
    _c_node : ctypes.POINTER
        Underlying C node pointer for interoperability with C backend.
    _python_backward_fn : Callable
        Python-specific backward function (if provided).

    Methods
    -------
    topo_sort() -> List[Node]
        Performs topological sorting starting from this node and returns
        an ordered list of nodes such that inputs are realized before outputs.

    realize(graph: List[Node]) -> None
        Executes the forward functions of nodes in topological order, updating
        their output tensors with computed values.

    backward(graph: List[Node]) -> None
        Executes backward passes for each node in reverse topological order.
        Propagates gradients through the graph by calling Python or C
        backward functions.
    """

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

                if node._c_node and not node._c_node.out:
                    node._c_node.out = node.out_tensor._c_tensor

    def backward(self, graph):
        for i, node in enumerate(graph):
            c_prev_array = (ctypes.POINTER(CTensor) * len(node.input_tensors))()
            for i, t in enumerate(node.input_tensors):
                c_prev_array[i] = t._c_tensor

            if isinstance(node._extras_obj, ctypes._SimpleCData):
                extras_to_pass = ctypes.byref(node._extras_obj)
            else:
                extras_to_pass = node._extras_obj

            if node._python_backward_fn:
                node._python_backward_fn(
                    node.out_tensor._c_tensor,
                    c_prev_array,
                    len(node.input_tensors),
                    extras_to_pass,
                )
            elif node.backward_fn:
                node.backward_fn(
                    node.out_tensor._c_tensor,
                    c_prev_array,
                    len(node.input_tensors),
                    extras_to_pass,
                )

            for input_tensor in node.input_tensors:
                if (
                    input_tensor.requires_grad
                    and hasattr(input_tensor, "_node")
                    and input_tensor._node
                ):
                    input_tensor._node.backward(graph[i + 1 :])


if __name__ == "__main__":
    from .tensor import Tensor

    def dummy_backward(*args):
        pass

    a = Tensor(shape=(1,), data=[1.0], requires_grad=True)
    b = Tensor(shape=(1,), data=[2.0], requires_grad=True)
    c = Tensor(shape=(1,), data=[3.0], requires_grad=True)
    d = Tensor(shape=(1,), data=[4.0], requires_grad=True)

    def dummy_forward(*args, **kwargs):
        # This dummy forward function will just return the out_tensor
        # In a real scenario, this would perform the actual operation
        return kwargs.get(
            "out_tensor_val"
        )  # Assuming out_tensor_val is passed in kwargs for simplicity

    node_c = Node(
        out_tensor=c,
        input_tensors=[a, b],
        forward_fn=dummy_forward,
        forward_args=(),
        forward_kwargs={"out_tensor_val": c},
        backward_fn=dummy_backward,
    )
    c._node = node_c

    node_d = Node(
        out_tensor=d,
        input_tensors=[c],
        forward_fn=dummy_forward,
        forward_args=(),
        forward_kwargs={"out_tensor_val": d},
        backward_fn=dummy_backward,
    )
    d._node = node_d

    print("Performing topological sort starting from node_d:")
    sorted_nodes = node_d.topo_sort()

    print(
        "Topologically sorted nodes (should be c, a, b, d or similar order of a,b before c, and c before d):"
    )
    for node in sorted_nodes:
        print(f"Node producing tensor with data: {node.out_tensor.data}")
