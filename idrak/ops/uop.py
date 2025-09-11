from __future__ import annotations
import ctypes
from .op import LazyOp
from idrak.idrak_bindings.ctypes_definitions import CTensor, ClipExtras
from idrak.idrak_bindings.c_wrapper_functions import c_relu, c_log, c_exp, c_abs, c_neg, c_relu_grad_op, c_log_grad_op, c_abs_grad_op, c_exp_grad_op, c_neg_grad_op, c_clip, c_clip_grad_op

class UOp(LazyOp):
    @staticmethod
    def calc_out_shape(a: Tensor, **kwargs) -> tuple[int, ...]:
        return a.shape

    def create_ctx_struct(self, *args, **kwargs):
        return None
    
class ReLU(UOp):
    @staticmethod
    def forward(out: "Tensor", a: "Tensor"): 
        c_relu(a._c_tensor, out._c_tensor)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras): c_relu_grad_op(out_ptr, prev_ptrs, n_prev, extras)

class Log(UOp):
    @staticmethod
    def forward(out: "Tensor", a: "Tensor"): 
        c_log(a._c_tensor, out._c_tensor)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras): c_log_grad_op(out_ptr, prev_ptrs, n_prev, extras)

class Exp(UOp):
    @staticmethod
    def forward(out: "Tensor", a: "Tensor"): 
        c_exp(a._c_tensor, out._c_tensor)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras): c_exp_grad_op(out_ptr, prev_ptrs, n_prev, extras)


class Abs(UOp):
    @staticmethod
    def forward(out: "Tensor", a: "Tensor"): 
        c_abs(a._c_tensor, out._c_tensor)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras): c_abs_grad_op(out_ptr, prev_ptrs, n_prev, extras)

class Neg(UOp):
    @staticmethod
    def forward(out: "Tensor", a: "Tensor"): 
        c_neg(a._c_tensor, out._c_tensor)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras): c_neg_grad_op(out_ptr, prev_ptrs, n_prev, extras)

class Clip(UOp):
    @staticmethod
    def forward(out: "Tensor", a: "Tensor", min_val: float, max_val: float):
        c_clip(a._c_tensor, min_val, max_val, out._c_tensor)

    @staticmethod
    def backward(out_ptr: ctypes.POINTER(CTensor), prev_ptrs: ctypes.POINTER(ctypes.POINTER(CTensor)), n_prev: int, extras): c_clip_grad_op(out_ptr, prev_ptrs, n_prev, extras)

    def create_ctx_struct(self, *args, **kwargs):
        min_val = kwargs["min_val"]
        max_val = kwargs["max_val"]
        clip_extras = ClipExtras(min_val=min_val, max_val=max_val)
        return ctypes.pointer(clip_extras)
