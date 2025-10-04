from __future__ import annotations
import ctypes
from .op import LazyOp
from axon.axon_bindings.ctypes_definitions import CTensor, ClipExtras
from axon.axon_bindings.c_wrapper_functions import get_op_function

class UOp(LazyOp):
    @classmethod
    def calc_out_shape(cls, a: Tensor, **kwargs) -> tuple[int, ...]:
        return a.shape
    
    @classmethod
    def create_ctx_struct(cls, *args, **kwargs) -> tuple[Dict[str, Any], Any]:
        return kwargs, None

    def forward(self, out: "Tensor", a: "Tensor", **kwargs):
        raise NotImplementedError("Subclasses must implement the forward method.")

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras: Any):
        raise NotImplementedError("Subclasses must implement the backward method.")


class ReLU(UOp):
    def forward(self, out: "Tensor", a: "Tensor", **kwargs):
        relu_op_func = get_op_function("relu", a.device)
        relu_op_func(a.c_tensor_ptr, out.c_tensor_ptr)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras):
        out_tensor_struct = out_ptr.contents
        relu_grad_op_func = get_op_function("relu_grad", out_tensor_struct.device)
        relu_grad_op_func(out_ptr, prev_ptrs, n_prev, extras)

class Log(UOp):
    def forward(self, out: "Tensor", a: "Tensor", **kwargs):
        log_op_func = get_op_function("log", a.device)
        log_op_func(a.c_tensor_ptr, out.c_tensor_ptr)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras):
        out_tensor_struct = out_ptr.contents
        log_grad_op_func = get_op_function("log_grad", out_tensor_struct.device)
        log_grad_op_func(out_ptr, prev_ptrs, n_prev, extras)

class Exp(UOp):
    def forward(self, out: "Tensor", a: "Tensor", **kwargs):
        exp_op_func = get_op_function("exp", a.device)
        exp_op_func(a.c_tensor_ptr, out.c_tensor_ptr)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras):
        out_tensor_struct = out_ptr.contents
        exp_grad_op_func = get_op_function("exp_grad", out_tensor_struct.device)
        exp_grad_op_func(out_ptr, prev_ptrs, n_prev, extras)


class Abs(UOp):
    def forward(self, out: "Tensor", a: "Tensor", **kwargs):
        abs_op_func = get_op_function("abs", a.device)
        abs_op_func(a.c_tensor_ptr, out.c_tensor_ptr)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras):
        out_tensor_struct = out_ptr.contents
        abs_grad_op_func = get_op_function("abs_grad", out_tensor_struct.device)
        abs_grad_op_func(out_ptr, prev_ptrs, n_prev, extras)

class Neg(UOp):
    def forward(self, out: "Tensor", a: "Tensor", **kwargs):
        neg_op_func = get_op_function("neg", a.device)
        neg_op_func(a.c_tensor_ptr, out.c_tensor_ptr)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras):
        out_tensor_struct = out_ptr.contents
        neg_grad_op_func = get_op_function("neg_grad", out_tensor_struct.device)
        neg_grad_op_func(out_ptr, prev_ptrs, n_prev, extras)

class Clip(UOp):
    @classmethod
    def create_ctx_struct(cls, *args, **kwargs) -> tuple[Dict[str, Any], Any]:
        min_val = kwargs.get("min_val")
        max_val = kwargs.get("max_val")

        if min_val is None or max_val is None:
            raise ValueError("Clip operation requires 'min_val' and 'max_val' keyword arguments.")

        clip_extras = ClipExtras(min_val=ctypes.c_float(min_val), max_val=ctypes.c_float(max_val))
        ctx = ctypes.pointer(clip_extras)

        forward_kwargs = {"min_val": min_val, "max_val": max_val}
        return forward_kwargs, ctypes.cast(ctx, ctypes.c_void_p)

    def forward(self, out: "Tensor", a: "Tensor", **kwargs):
        min_val = kwargs["min_val"]
        max_val = kwargs["max_val"]
        clip_op_func = get_op_function("clip", a.device)
        clip_op_func(a.c_tensor_ptr, out.c_tensor_ptr, min_val, max_val)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras):
        out_tensor_struct = out_ptr.contents
        clip_grad_op_func = get_op_function("clip_grad", out_tensor_struct.device)
        clip_grad_op_func(out_ptr, prev_ptrs, n_prev, extras)


