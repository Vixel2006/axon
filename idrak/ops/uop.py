from __future__ import annotations
import ctypes
from .op import LazyOp
from idrak.idrak_bindings.ctypes_definitions import CTensor, ClipExtras
from idrak.idrak_bindings.c_wrapper_functions import c_relu, c_log, c_exp, c_abs, c_neg, c_relu_grad_op, c_log_grad_op, c_abs_grad_op, c_exp_grad_op, c_neg_grad_op, c_clip, c_clip_grad_op

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
        c_relu(a.c_tensor_ptr, out.c_tensor_ptr)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras):
        c_relu_grad_op(out_ptr, prev_ptrs, n_prev, extras)

class Log(UOp):
    def forward(self, out: "Tensor", a: "Tensor", **kwargs):
        c_log(a.c_tensor_ptr, out.c_tensor_ptr)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras):
        c_log_grad_op(out_ptr, prev_ptrs, n_prev, extras)

class Exp(UOp):
    def forward(self, out: "Tensor", a: "Tensor", **kwargs):
        c_exp(a.c_tensor_ptr, out.c_tensor_ptr)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras):
        c_exp_grad_op(out_ptr, prev_ptrs, n_prev, extras)


class Abs(UOp):
    def forward(self, out: "Tensor", a: "Tensor", **kwargs):
        c_abs(a.c_tensor_ptr, out.c_tensor_ptr)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras):
        c_abs_grad_op(out_ptr, prev_ptrs, n_prev, extras)

class Neg(UOp):

    def forward(self, out: "Tensor", a: "Tensor", **kwargs):
        c_neg(a.c_tensor_ptr, out.c_tensor_ptr)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras):
        c_neg_grad_op(out_ptr, prev_ptrs, n_prev, extras)

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
        c_clip(a.c_tensor_ptr, out.c_tensor_ptr, min_val, max_val)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras):
        c_clip_grad_op(out_ptr, prev_ptrs, n_prev, extras)


