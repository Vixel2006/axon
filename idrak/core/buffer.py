from __future__ import annotations
from os import wait

from numpy import where
from idrak.core.tensor import Tensor
from idrak.ops.op import LazyOp
from typing import List, Dict, Any
import ctypes
from idrak.idrak_bindings.ctypes_definitions import CTensor
from idrak.idrak_bindings.c_wrapper_functions import c_gmalloc

class LazyBuffer:
    def __init__(self, out: 'Tensor', op: 'LazyOp', prev: List['Tensor'], forward_kwargs: Dict[str, Any], backward_ctx: Any = None):
        self.out = out
        self.op = op
        self.prev = prev
        self.forward_kwargs = forward_kwargs
        self.backward_ctx = backward_ctx
        self._realized = False
        self._topo_sorted = None

    def topo_sort(self) -> List[LazyBuffer]:
        if self._topo_sorted is not None:
            return self._topo_sorted

        visited = set()
        temp_visited = set()
        result = []

        def visit(buffer):
            if buffer in temp_visited:
                raise RuntimeError("Circular dependency detected in computation graph")
            if buffer in visited:
                return

            temp_visited.add(buffer)

            for tensor in buffer.prev:
                if hasattr(tensor, '_lazy_buffer') and tensor._lazy_buffer:
                    visit(tensor._lazy_buffer)

            temp_visited.remove(buffer)
            visited.add(buffer)
            result.append(buffer)

        visit(self)
        self._topo_sorted = result
        return result

    def realize(self) -> Tensor:
        if self._realized:
            return self.out

        execution_order = self.topo_sort()

        for buffer in execution_order:
            if not buffer._realized:
                inputs_for_op_tensors = []
                for tensor in buffer.prev:
                    if hasattr(tensor, '_lazy_buffer') and tensor._lazy_buffer:
                        tensor._lazy_buffer.realize()
                    inputs_for_op_tensors.append(tensor)

                buffer.op.forward(buffer.out, *inputs_for_op_tensors, **buffer.forward_kwargs)

                buffer._realized = True

        return self.out

    def backward(self):
        self.realize()

        execution_order = self.topo_sort()[::-1]

        for i, buffer in enumerate(execution_order):
            if i == 0:
                c_gmalloc(buffer.out.c_tensor_ptr, ctypes.c_float(1.0))
            out_grad_ptr = buffer.out.c_tensor_ptr
            in_grad_ptrs_type = ctypes.POINTER(CTensor) * len(buffer.prev)
            in_grad_ptrs = in_grad_ptrs_type(*(t.c_tensor_ptr for t in buffer.prev))
            in_grad_ptrs_ptr = ctypes.cast(in_grad_ptrs, ctypes.POINTER(CTensor))
            buffer.op.backward(out_grad_ptr, in_grad_ptrs_ptr, len(buffer.prev), buffer.backward_ctx)
