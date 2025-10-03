from __future__ import annotations
from os import wait

from numpy import where
from fajr.core.tensor import Tensor
from fajr.ops.op import LazyOp
from typing import List, Dict, Any
import ctypes
from fajr.fajr_bindings.ctypes_definitions import CTensor, CStorage
from fajr.fajr_bindings.c_wrapper_functions import c_gmalloc

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
            if not buffer.out.requires_grad:
                continue

            if i == 0:
                c_gmalloc(buffer.out.c_tensor_ptr, ctypes.c_float(1.0))
            
            out_grad_storage_ptr = buffer.out.c_tensor_ptr

            if not out_grad_storage_ptr:
                continue

            prev_grad_storage_ptrs = []
            for tensor in buffer.prev:
                if tensor.requires_grad:
                    if not tensor.c_tensor_ptr.contents.grad:
                        raise RuntimeError(f"Gradient storage not allocated for input tensor with shape {tensor.shape} and requires_grad=True")
                    prev_grad_storage_ptrs.append(tensor.c_tensor_ptr)
            
            if not prev_grad_storage_ptrs and i != 0:
                continue

            in_grad_ptrs_type = ctypes.POINTER(CTensor) * len(prev_grad_storage_ptrs)
            in_grad_ptrs_array = in_grad_ptrs_type(*prev_grad_storage_ptrs)
            
            in_grad_ptrs_ptr = ctypes.cast(in_grad_ptrs_array, ctypes.POINTER(ctypes.POINTER(CTensor)))
            
            buffer.op.backward(out_grad_storage_ptr, in_grad_ptrs_ptr, len(prev_grad_storage_ptrs), buffer.backward_ctx)

