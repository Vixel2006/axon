import ctypes
from os import wait
import numpy as np
from .c_library_loader import tensor_lib
from .c_function_signatures import *
from .ctypes_definitions import CTensor, CStorage, CDevice


# TODO: add type annotations for ease of development

if tensor_lib:
    def c_numel(shape, ndim):
        if ndim == 0:
            return 1
        c_shape = (ctypes.c_int * ndim)(*shape)
        return tensor_lib.numel(c_shape, ndim)

    def c_compute_strides(shape, ndim):
        if ndim == 0:
            return None
        c_shape = (ctypes.c_int * ndim)(*shape)
        return tensor_lib.compute_strides(c_shape, ndim)

    def c_smalloc(data, size):
        return tensor_lib.smalloc(data, size)

    def c_sfree(storage):
        return tensor_lib.sfree(storage)

    def c_gmalloc(grad_ptr, grad):
        return tensor_lib.gmalloc(grad_ptr, grad)

    def c_gfree(tensor_ptr):
        return tensor_lib.gfree(tensor_ptr)

    def c_tmalloc(shape, ndim, device, requires_grad):
        c_shape = (ctypes.c_int * ndim)(*shape)
        return tensor_lib.tmalloc(c_shape, ndim, device, requires_grad)

    def c_tfree(tensor_ptr):
        tensor_lib.tfree(tensor_ptr)

    def c_zeros(tensor_ptr):
        return tensor_lib.zeros(tensor_ptr)

    def c_ones(tensor_ptr):
        return tensor_lib.ones(tensor_ptr)

    def c_randn(tensor_ptr):
        return tensor_lib.randn(tensor_ptr)

    def c_uniform(tensor_ptr, low, high):
        return tensor_lib.uniform(tensor_ptr, low, high)

    def c_add_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.add_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_sub_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.sub_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_rsub_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.rsub_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_mul_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.mul_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_pow_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.pow_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_div_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.div_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_rdiv_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.rdiv_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_matmul_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.matmul_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_conv_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.conv2d_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_relu_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.relu_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_log_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.log_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_exp_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.exp_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_softmax_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.softmax_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_abs_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.abs_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_neg_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.neg_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_clip_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.clip_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_sum_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.sum_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_mean_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.mean_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_max_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.max_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_sum_full_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.sum_full_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_mean_full_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.mean_full_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_max_full_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.max_full_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_stack_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.stack_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_concat_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.concat_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_dot_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.dot_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    # Unary operations wrappers
    def c_relu(in_tensor_ptr, out_tensor_ptr):
        tensor_lib.relu_op(in_tensor_ptr, out_tensor_ptr)

    def c_log(in_tensor_ptr, out_tensor_ptr):
        tensor_lib.log_op(in_tensor_ptr, out_tensor_ptr)

    def c_exp(in_tensor_ptr, out_tensor_ptr):
        tensor_lib.exp_op(in_tensor_ptr, out_tensor_ptr)

    def c_softmax(in_tensor_ptr, out_tensor_ptr):
        tensor_lib.softmax_op(in_tensor_ptr, out_tensor_ptr)

    def c_abs(in_tensor_ptr, out_tensor_ptr):
        tensor_lib.abs_op(in_tensor_ptr, out_tensor_ptr)

    def c_neg(in_tensor_ptr, out_tensor_ptr):
        tensor_lib.neg_op(in_tensor_ptr, out_tensor_ptr)

    def c_clip(in_tensor_ptr, min_val, max_val, out_tensor_ptr):
        tensor_lib.clip_op(in_tensor_ptr, min_val, max_val, out_tensor_ptr)

    def c_tanh(in_tensor_ptr, out_tensor_ptr):
        tensor_lib.tanh_op(in_tensor_ptr, out_tensor_ptr)

    def c_sigmoid(in_tensor_ptr, out_tensor_ptr):
        tensor_lib.sigmoid_op(in_tensor_ptr, out_tensor_ptr)

    # Reduction operations wrappers
    def c_sum(in_tensor_ptr, out_tensor_ptr, axis, keepdim):
        tensor_lib.sum_op(in_tensor_ptr, out_tensor_ptr, axis, keepdim)

    def c_mean(in_tensor_ptr, out_tensor_ptr, axis, keepdim):
        tensor_lib.mean_op(in_tensor_ptr, out_tensor_ptr, axis, keepdim)

    def c_max(in_tensor_ptr, out_tensor_ptr, axis, keepdim):
        tensor_lib.max_op(in_tensor_ptr, out_tensor_ptr, axis, keepdim)

    def c_sum_full(in_tensor_ptr, out_tensor_ptr):
        tensor_lib.sum_full_op(in_tensor_ptr, out_tensor_ptr)

    def c_mean_full(in_tensor_ptr, out_tensor_ptr):
        tensor_lib.mean_full_op(in_tensor_ptr, out_tensor_ptr)

    def c_max_full(in_tensor_ptr, out_tensor_ptr):
        tensor_lib.max_full_op(in_tensor_ptr, out_tensor_ptr)

    # Binary operations wrappers
    def c_add(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr):
        tensor_lib.add_op(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr)

    def c_sub(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr):
        tensor_lib.sub_op(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr)

    def c_mul(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr):
        tensor_lib.mul_op(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr)

    def c_div(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr):
        tensor_lib.div_op(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr)

    def c_matmul(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr, N, K, P):
        tensor_lib.matmul_op(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr, N, K, P)

    def c_conv(
        a_tensor_ptr, b_tensor_ptr, out_tensor_ptr, kernel_size, stride, padding
    ):
        c_kernel_size = (ctypes.c_int * len(kernel_size))(*kernel_size)
        c_stride = (ctypes.c_int * len(stride))(*stride)
        tensor_lib.conv2d_op(
            a_tensor_ptr, b_tensor_ptr, out_tensor_ptr, c_kernel_size, c_stride, padding
        )

    def c_dot(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr):
        tensor_lib.dot_op(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr)

    # Binary operations with scalars wrappers
    def c_add_scalar(a_tensor_ptr, b, out_tensor_ptr):
        tensor_lib.add_scalar_op(a_tensor_ptr, b, out_tensor_ptr)

    def c_sub_scalar(a_tensor_ptr, b, out_tensor_ptr):
        tensor_lib.sub_scalar_op(a_tensor_ptr, b, out_tensor_ptr)

    def c_rsub_scalar(a, b_tensor_ptr, out_tensor_ptr):
        tensor_lib.rsub_scalar_op(a, b_tensor_ptr, out_tensor_ptr)

    def c_mul_scalar(a_tensor_ptr, b, out_tensor_ptr):
        tensor_lib.mul_scalar_op(a_tensor_ptr, b, out_tensor_ptr)

    def c_div_scalar(a_tensor_ptr, b, out_tensor_ptr):
        tensor_lib.div_scalar_op(a_tensor_ptr, b, out_tensor_ptr)

    def c_rdiv_scalar(a, b_tensor_ptr, out_tensor_ptr):
        tensor_lib.rdiv_scalar_op(a, b_tensor_ptr, out_tensor_ptr)

    def c_pow_scalar(a_tensor_ptr, b, out_tensor_ptr):
        tensor_lib.pow_scalar_op(a_tensor_ptr, b, out_tensor_ptr)

    def c_pow(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr):
        tensor_lib.pow_op(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr)

    # Movement operations wrappers
    def c_view(in_tensor_ptr, out_tensor_ptr, shape, ndim):
        c_shape = (ctypes.c_int * ndim)(*shape)
        tensor_lib.view_op(in_tensor_ptr, out_tensor_ptr, c_shape, ndim)

    def c_unsqueeze(in_tensor_ptr, out_tensor_ptr, dim):
        tensor_lib.unsqueeze_op(in_tensor_ptr, out_tensor_ptr, dim)

    def c_squeeze(in_tensor_ptr, out_tensor_ptr, dim):
        tensor_lib.squeeze_op(in_tensor_ptr, out_tensor_ptr, dim)

    def c_transpose(in_tensor_ptr, out_tensor_ptr, n, m):
        tensor_lib.transpose_op(in_tensor_ptr, out_tensor_ptr, n, m)

    def c_expand(in_tensor_ptr, out_tensor_ptr, shape):
        ndim = len(shape)
        c_shape = (ctypes.c_int * ndim)(*shape)
        tensor_lib.expand_op(in_tensor_ptr, out_tensor_ptr, c_shape)

    def c_broadcast(in_tensor_ptr, out_tensor_ptr, shape, ndim):
        c_shape = (ctypes.c_int * ndim)(*shape)
        tensor_lib.broadcast_op(in_tensor_ptr, out_tensor_ptr, ndim, c_shape)

    def c_concat(in_tensor_ptrs, out_tensor_ptr, num_tensors, axis):
        tensor_lib.concat_op(in_tensor_ptrs, out_tensor_ptr, num_tensors, axis)

    # Optimizers
    def c_sgd(params, num_params, lr):
        in_param_ptrs = (ctypes.POINTER(CTensor) * num_params)(*params)
        tensor_lib.sgd(in_param_ptrs, num_params, lr)

    def c_adam(params, m_estimates, v_estimates, num_params, time_step, lr, beta1, beta2, epsilon):
        in_param_ptrs = (ctypes.POINTER(CTensor) * num_params)(*params)
        m_estimates_ptrs = (ctypes.POINTER(CTensor) * num_params)(*m_estimates)
        v_estimates_ptrs = (ctypes.POINTER(CTensor) * num_params)(*v_estimates)
        tensor_lib.adam(in_param_ptrs, m_estimates_ptrs, v_estimates_ptrs, num_params, time_step, lr, beta1, beta2, epsilon)


    def c_zero_grad(params, num_params):
        in_param_ptrs = (ctypes.POINTER(CTensor) * num_params)(*params)
        tensor_lib.zero_grad(in_param_ptrs, num_params)


    def c_set_debug_mode(enable):
        tensor_lib.idrak_set_debug_mode(ctypes.c_int(enable))
else:
    # WARNING: We should make this raise an error
    def c_numel(shape, ndim): pass
    
    def c_compute_strides(shape, ndim): pass

    def c_smalloc(data, size): pass

    def c_sfree(storage): pass

    def c_gmalloc(grad_ptr, grad): pass

    def c_gfree(tensor_ptr): pass

    def c_tmalloc(shape, ndim, device, requires_grad): pass

    def c_tfree(tensor_ptr): pass

    def c_zeros(tensor_ptr): pass

    def c_ones(tensor_ptr): pass

    def c_randn(tensor_ptr): pass

    def c_uniform(tensor_ptr, low, high): pass

    def c_add_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras): pass

    def c_sub_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras): pass

    def c_rsub_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras): pass

    def c_mul_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras): pass

    def c_pow_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras): pass

    def c_div_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras): pass

    def c_rdiv_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras): pass

    def c_matmul_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras): pass

    def c_conv_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras): pass

    def c_relu_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras): pass

    def c_log_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras): pass

    def c_exp_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras): pass

    def c_softmax_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras): pass

    def c_abs_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras): pass

    def c_neg_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras): pass

    def c_clip_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras): pass

    def c_sum_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras): pass

    def c_mean_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras): pass

    def c_max_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras): pass

    def c_sum_full_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras): pass

    def c_mean_full_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras): pass

    def c_max_full_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras): pass

    def c_stack_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras): pass

    def c_concat_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras): pass

    def c_dot_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras): pass

    # Unary operations wrappers
    def c_relu(in_tensor_ptr, out_tensor_ptr): pass

    def c_log(in_tensor_ptr, out_tensor_ptr): pass

    def c_exp(in_tensor_ptr, out_tensor_ptr): pass

    def c_softmax(in_tensor_ptr, out_tensor_ptr): pass

    def c_abs(in_tensor_ptr, out_tensor_ptr): pass

    def c_neg(in_tensor_ptr, out_tensor_ptr): pass

    def c_clip(in_tensor_ptr, min_val, max_val, out_tensor_ptr): pass

    def c_tanh(in_tensor_ptr, out_tensor_ptr): pass

    def c_sigmoid(in_tensor_ptr, out_tensor_ptr): pass

    # Reduction operations wrappers
    def c_sum(in_tensor_ptr, out_tensor_ptr, axis, keepdim): pass

    def c_mean(in_tensor_ptr, out_tensor_ptr, axis, keepdim): pass

    def c_max(in_tensor_ptr, out_tensor_ptr, axis, keepdim): pass

    def c_sum_full(in_tensor_ptr, out_tensor_ptr): pass

    def c_mean_full(in_tensor_ptr, out_tensor_ptr): pass

    def c_max_full(in_tensor_ptr, out_tensor_ptr): pass

    # Binary operations wrappers
    def c_add(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr): pass

    def c_sub(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr): pass

    def c_mul(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr): pass

    def c_div(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr): pass

    def c_matmul(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr, N, K, P): pass

    def c_conv(
        a_tensor_ptr, b_tensor_ptr, out_tensor_ptr, kernel_size, stride, padding
    ): pass

    def c_dot(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr): pass

    # Binary operations with scalars wrappers
    def c_add_scalar(a_tensor_ptr, b, out_tensor_ptr): pass

    def c_sub_scalar(a_tensor_ptr, b, out_tensor_ptr): pass

    def c_rsub_scalar(a, b_tensor_ptr, out_tensor_ptr): pass

    def c_mul_scalar(a_tensor_ptr, b, out_tensor_ptr): pass

    def c_div_scalar(a_tensor_ptr, b, out_tensor_ptr): pass

    def c_rdiv_scalar(a, b_tensor_ptr, out_tensor_ptr): pass

    def c_pow_scalar(a_tensor_ptr, b, out_tensor_ptr): pass

    def c_pow(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr): pass

    # Movement operations wrappers
    def c_view(in_tensor_ptr, out_tensor_ptr, shape, ndim): pass

    def c_unsqueeze(in_tensor_ptr, out_tensor_ptr, dim): pass

    def c_squeeze(in_tensor_ptr, out_tensor_ptr, dim): pass

    def c_transpose(in_tensor_ptr, out_tensor_ptr, n, m): pass

    def c_expand(in_tensor_ptr, out_tensor_ptr, shape): pass

    def c_broadcast(in_tensor_ptr, out_tensor_ptr, ndim, shape): pass

    def c_concat(in_tensors, out_tensor_ptr, num_tensors, axis): pass

    # Optimizers
    def c_sgd(params, num_params, lr): pass 

    def c_adam(params, m_estimates, v_estimates, num_params, time_step, lr, beta1, beta2, epsilon): pass

    def c_zero_grad(params, num_params): pass

    def c_set_debug_mode(enable): pass
