import ctypes
from os import wait
import numpy as np
from .c_library_loader import tensor_lib
from .c_function_signatures import *
from .ctypes_definitions import CTensor, CStorage, CDevice

# TODO: add type annotations for ease of development

_ops_map = {}

def _register_op(op_name, cpu_func, cuda_func=None):
    _ops_map[op_name] = {"cpu": cpu_func, "cuda": cuda_func}

def get_op_function(op_name, device):
    from axon.core.device import Device
    if op_name not in _ops_map:
        raise ValueError(f"Operation '{op_name}' not registered.")
    
    device_type_id = -1
    if isinstance(device, Device):
        device_type_id = device.type_id
    elif isinstance(device, ctypes.POINTER(CDevice)):
        device_type_id = device.contents.type
    elif isinstance(device, CDevice):
        device_type_id = device.type
    else:
        raise TypeError(f"Unknown device type: {type(device)}")
    
    if device_type_id == 0:
        device_str = "cpu"
    elif device_type_id == 1:
        device_str = "cuda"
    else:
        raise ValueError(f"Unknown device type id: {device_type_id}")
        
    func = _ops_map[op_name].get(device_str)
    if func is None:
        raise ValueError(f"Operation '{op_name}' not available for device '{device_str}'.")
    return func


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

    def c_dmalloc(device_type, device_index):
        return tensor_lib.dmalloc(ctypes.c_int(device_type), ctypes.c_int(device_index))

    def c_dfree(device):
        return tensor_lib.dfree(device)

    def c_smalloc(data, size, device: "Device"):
        c_device = CDevice(type=device.type_id, index=device.index)
        return tensor_lib.smalloc(data, size, ctypes.byref(c_device))

    def c_sfree(storage, device: "Device"):
        c_device = CDevice(type=device.type_id, index=device.index)
        return tensor_lib.sfree(storage, ctypes.byref(c_device))

    def c_gmalloc(grad_ptr, grad):
        return tensor_lib.gmalloc(grad_ptr, grad)

    def c_tmalloc(shape, ndim, device, requires_grad):
        c_shape = (ctypes.c_int * ndim)(*shape)
        return tensor_lib.tmalloc(c_shape, ndim, device.c_device_ptr, requires_grad)

    def c_tfree(tensor_ptr):
        tensor_lib.tfree(tensor_ptr)

    def c_copy_storage_to_host(storage_ptr, device: "Device", size, shape, strides, ndim, host_buffer):
        c_device = CDevice(type=device.type_id, index=device.index)
        c_shape = (ctypes.c_int * ndim)(*shape)
        c_strides = (ctypes.c_int * ndim)(*strides)
        tensor_lib.copy_storage_to_host(storage_ptr, ctypes.byref(c_device), size, c_shape, c_strides, ndim, host_buffer)

    def c_zeros(tensor_ptr):
        return tensor_lib.zeros(tensor_ptr)

    def c_ones(tensor_ptr):
        return tensor_lib.ones(tensor_ptr)

    def c_randn(tensor_ptr):
        return tensor_lib.randn(tensor_ptr)

    def c_uniform(tensor_ptr, low, high):
        return tensor_lib.uniform(tensor_ptr, low, high)

    def c_from_data(tensor_ptr, data):
        return tensor_lib.from_data(tensor_ptr, data)

    def c_borrow(out_tensor_ptr, storage_ptr, grad_storage_ptr):
        return tensor_lib.borrow(out_tensor_ptr, storage_ptr, grad_storage_ptr)

    def c_to(tensor_ptr, device: "Device"):
        return tensor_lib.to(tensor_ptr, device)

    # Gradient operations wrappers
    def c_add_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.add_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_add_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.add_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_sub_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.sub_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_sub_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.sub_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_rsub_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.rsub_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_rsub_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.rsub_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_mul_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.mul_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_mul_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.mul_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_pow_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.pow_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_pow_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.pow_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_div_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.div_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_div_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.div_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_rdiv_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.rdiv_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_rdiv_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.rdiv_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_matmul_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.matmul_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_matmul_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.matmul_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_conv_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.conv2d_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_conv_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.conv2d_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_dot_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.dot_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_dot_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.dot_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_relu_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.relu_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_relu_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.relu_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_log_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.log_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_log_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.log_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_exp_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.exp_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_exp_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.exp_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_softmax_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.softmax_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_softmax_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.softmax_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_abs_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.abs_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_abs_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.abs_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_neg_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.neg_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_neg_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.neg_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_clip_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.clip_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_clip_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.clip_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_sum_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.sum_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_sum_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.sum_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_mean_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.mean_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_mean_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.mean_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_max_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.max_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_max_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.max_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_sum_full_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.sum_full_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_sum_full_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.sum_full_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_mean_full_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.mean_full_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_mean_full_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.mean_full_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_max_full_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.max_full_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_max_full_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.max_full_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_stack_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.stack_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_stack_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.stack_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_concat_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.concat_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_concat_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.concat_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_dot_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.dot_grad_op_cpu(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_dot_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.dot_grad_op_cuda(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    # Unary operations wrappers
    def c_relu_cpu(in_tensor_ptr, out_tensor_ptr):
        tensor_lib.relu_op_cpu(in_tensor_ptr, out_tensor_ptr)

    def c_relu_cuda(in_tensor_ptr, out_tensor_ptr):
        tensor_lib.relu_op_cuda(in_tensor_ptr, out_tensor_ptr)

    def c_log_cpu(in_tensor_ptr, out_tensor_ptr):
        tensor_lib.log_op_cpu(in_tensor_ptr, out_tensor_ptr)

    def c_log_cuda(in_tensor_ptr, out_tensor_ptr):
        tensor_lib.log_op_cuda(in_tensor_ptr, out_tensor_ptr)

    def c_exp_cpu(in_tensor_ptr, out_tensor_ptr):
        tensor_lib.exp_op_cpu(in_tensor_ptr, out_tensor_ptr)

    def c_exp_cuda(in_tensor_ptr, out_tensor_ptr):
        tensor_lib.exp_op_cuda(in_tensor_ptr, out_tensor_ptr)

    def c_softmax_cpu(in_tensor_ptr, out_tensor_ptr):
        tensor_lib.softmax_op_cpu(in_tensor_ptr, out_tensor_ptr)

    def c_softmax_cuda(in_tensor_ptr, out_tensor_ptr):
        tensor_lib.softmax_op_cuda(in_tensor_ptr, out_tensor_ptr)

    def c_abs_cpu(in_tensor_ptr, out_tensor_ptr):
        tensor_lib.abs_op_cpu(in_tensor_ptr, out_tensor_ptr)

    def c_abs_cuda(in_tensor_ptr, out_tensor_ptr):
        tensor_lib.abs_op_cuda(in_tensor_ptr, out_tensor_ptr)

    def c_neg_cpu(in_tensor_ptr, out_tensor_ptr):
        tensor_lib.neg_op_cpu(in_tensor_ptr, out_tensor_ptr)

    def c_neg_cuda(in_tensor_ptr, out_tensor_ptr):
        tensor_lib.neg_op_cuda(in_tensor_ptr, out_tensor_ptr)

    def c_clip_cpu(in_tensor_ptr, out_tensor_ptr, min_val, max_val):
        tensor_lib.clip_op_cpu(in_tensor_ptr,out_tensor_ptr, min_val, max_val)

    def c_clip_cuda(in_tensor_ptr, out_tensor_ptr, min_val, max_val):
        tensor_lib.clip_op_cuda(in_tensor_ptr,out_tensor_ptr, min_val, max_val)

    # Reduction operations wrappers
    def c_sum_cpu(in_tensor_ptr, out_tensor_ptr, axis, keepdim):
        tensor_lib.sum_op_cpu(in_tensor_ptr, out_tensor_ptr, axis, keepdim)

    def c_sum_cuda(in_tensor_ptr, out_tensor_ptr, axis, keepdim):
        tensor_lib.sum_op_cuda(in_tensor_ptr, out_tensor_ptr, axis, keepdim)

    def c_mean_cpu(in_tensor_ptr, out_tensor_ptr, axis, keepdim):
        tensor_lib.mean_op_cpu(in_tensor_ptr, out_tensor_ptr, axis, keepdim)

    def c_mean_cuda(in_tensor_ptr, out_tensor_ptr, axis, keepdim):
        tensor_lib.mean_op_cuda(in_tensor_ptr, out_tensor_ptr, axis, keepdim)

    def c_max_cpu(in_tensor_ptr, out_tensor_ptr, axis, keepdim):
        tensor_lib.max_op_cpu(in_tensor_ptr, out_tensor_ptr, axis, keepdim)

    def c_max_cuda(in_tensor_ptr, out_tensor_ptr, axis, keepdim):
        tensor_lib.max_op_cuda(in_tensor_ptr, out_tensor_ptr, axis, keepdim)

    def c_sum_full_cpu(in_tensor_ptr, out_tensor_ptr):
        tensor_lib.sum_full_op_cpu(in_tensor_ptr, out_tensor_ptr)

    def c_sum_full_cuda(in_tensor_ptr, out_tensor_ptr):
        tensor_lib.sum_full_op_cuda(in_tensor_ptr, out_tensor_ptr)

    def c_mean_full_cpu(in_tensor_ptr, out_tensor_ptr):
        tensor_lib.mean_full_op_cpu(in_tensor_ptr, out_tensor_ptr)

    def c_mean_full_cuda(in_tensor_ptr, out_tensor_ptr):
        tensor_lib.mean_full_op_cuda(in_tensor_ptr, out_tensor_ptr)

    def c_max_full_cpu(in_tensor_ptr, out_tensor_ptr):
        tensor_lib.max_full_op_cpu(in_tensor_ptr, out_tensor_ptr)

    def c_max_full_cuda(in_tensor_ptr, out_tensor_ptr):
        tensor_lib.max_full_op_cuda(in_tensor_ptr, out_tensor_ptr)

    # Binary operations wrappers
    def c_add_cpu(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr):
        tensor_lib.add_op_cpu(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr)

    def c_add_cuda(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr):
        tensor_lib.add_op_cuda(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr)

    def c_sub_cpu(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr):
        tensor_lib.sub_op_cpu(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr)

    def c_sub_cuda(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr):
        tensor_lib.sub_op_cuda(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr)

    def c_mul_cpu(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr):
        tensor_lib.mul_op_cpu(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr)

    def c_mul_cuda(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr):
        tensor_lib.mul_op_cuda(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr)

    def c_div_cpu(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr):
        tensor_lib.div_op_cpu(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr)

    def c_div_cuda(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr):
        tensor_lib.div_op_cuda(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr)

    def c_matmul_cpu(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr, N, K, P):
        tensor_lib.matmul_op_cpu(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr, N, K, P)

    def c_matmul_cuda(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr, N, K, P):
        tensor_lib.matmul_op_cuda(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr, N, K, P)

    def c_conv_cpu(
        a_tensor_ptr, b_tensor_ptr, out_tensor_ptr, kernel_size, stride, padding
    ):
        c_kernel_size = (ctypes.c_int * len(kernel_size))(*kernel_size)
        c_stride = (ctypes.c_int * len(stride))(*stride)
        tensor_lib.conv2d_op_cpu(
            a_tensor_ptr, b_tensor_ptr, out_tensor_ptr, c_kernel_size, c_stride, padding
        )

    def c_conv_cuda(
        a_tensor_ptr, b_tensor_ptr, out_tensor_ptr, kernel_size, stride, padding
    ):
        c_kernel_size = (ctypes.c_int * len(kernel_size))(*kernel_size)
        c_stride = (ctypes.c_int * len(stride))(*stride)
        tensor_lib.conv2d_op_cuda(
            a_tensor_ptr, b_tensor_ptr, out_tensor_ptr, c_kernel_size, c_stride, padding
        )

    def c_dot_cpu(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr):
        tensor_lib.dot_op_cpu(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr)

    def c_dot_cuda(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr):
        tensor_lib.dot_op_cuda(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr)

    # Binary operations with scalars wrappers
    def c_add_scalar_cpu(a_tensor_ptr, b, out_tensor_ptr):
        tensor_lib.add_scalar_op_cpu(a_tensor_ptr, b, out_tensor_ptr)

    def c_add_scalar_cuda(a_tensor_ptr, b, out_tensor_ptr):
        tensor_lib.add_scalar_op_cuda(a_tensor_ptr, b, out_tensor_ptr)

    def c_sub_scalar_cpu(a_tensor_ptr, b, out_tensor_ptr):
        tensor_lib.sub_scalar_op_cpu(a_tensor_ptr, b, out_tensor_ptr)

    def c_sub_scalar_cuda(a_tensor_ptr, b, out_tensor_ptr):
        tensor_lib.sub_scalar_op_cuda(a_tensor_ptr, b, out_tensor_ptr)

    def c_rsub_scalar_cpu(a, b_tensor_ptr, out_tensor_ptr):
        tensor_lib.rsub_scalar_op_cpu(a, b_tensor_ptr, out_tensor_ptr)

    def c_rsub_scalar_cuda(a, b_tensor_ptr, out_tensor_ptr):
        tensor_lib.rsub_scalar_op_cuda(a, b_tensor_ptr, out_tensor_ptr)

    def c_mul_scalar_cpu(a_tensor_ptr, b, out_tensor_ptr):
        tensor_lib.mul_scalar_op_cpu(a_tensor_ptr, b, out_tensor_ptr)

    def c_mul_scalar_cuda(a_tensor_ptr, b, out_tensor_ptr):
        tensor_lib.mul_scalar_op_cuda(a_tensor_ptr, b, out_tensor_ptr)

    def c_div_scalar_cpu(a_tensor_ptr, b, out_tensor_ptr):
        tensor_lib.div_scalar_op_cpu(a_tensor_ptr, b, out_tensor_ptr)

    def c_div_scalar_cuda(a_tensor_ptr, b, out_tensor_ptr):
        tensor_lib.div_scalar_op_cuda(a_tensor_ptr, b, out_tensor_ptr)

    def c_rdiv_scalar_cpu(a, b_tensor_ptr, out_tensor_ptr):
        tensor_lib.rdiv_scalar_op_cpu(a, b_tensor_ptr, out_tensor_ptr)

    def c_rdiv_scalar_cuda(a, b_tensor_ptr, out_tensor_ptr):
        tensor_lib.rdiv_scalar_op_cuda(a, b_tensor_ptr, out_tensor_ptr)

    def c_pow_scalar_cpu(a_tensor_ptr, b, out_tensor_ptr):
        tensor_lib.pow_scalar_op_cpu(a_tensor_ptr, b, out_tensor_ptr)

    def c_pow_scalar_cuda(a_tensor_ptr, b, out_tensor_ptr):
        tensor_lib.pow_scalar_op_cuda(a_tensor_ptr, b, out_tensor_ptr)

    def c_pow_cpu(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr):
        tensor_lib.pow_op_cpu(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr)

    def c_pow_cuda(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr):
        tensor_lib.pow_op_cuda(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr)

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

    def c_concat_cpu(in_tensor_ptrs, out_tensor_ptr, num_tensors, axis):
        tensor_lib.concat_op_cpu(in_tensor_ptrs, out_tensor_ptr, num_tensors, axis)

    def c_concat_cuda(in_tensor_ptrs, out_tensor_ptr, num_tensors, axis):
        tensor_lib.concat_op_cuda(in_tensor_ptrs, out_tensor_ptr, num_tensors, axis)
    
    # Optimizers
    def c_sgd_cpu(params, num_params, lr):
        in_param_ptrs = (ctypes.POINTER(CTensor) * num_params)(*params)
        tensor_lib.sgd_cpu(in_param_ptrs, num_params, lr)
    
    def c_sgd_cuda(params, num_params, lr):
        in_param_ptrs = (ctypes.POINTER(CTensor) * num_params)(*params)
        tensor_lib.sgd_cuda(in_param_ptrs, num_params, lr)

    def c_adam_cpu(params, m_estimates, v_estimates, num_params, time_step, lr, beta1, beta2, epsilon):
        in_param_ptrs = (ctypes.POINTER(CTensor) * num_params)(*params)
        m_estimates_ptrs = (ctypes.POINTER(CTensor) * num_params)(*m_estimates)
        v_estimates_ptrs = (ctypes.POINTER(CTensor) * num_params)(*v_estimates)
        tensor_lib.adam_cpu(in_param_ptrs, m_estimates_ptrs, v_estimates_ptrs, num_params, time_step, lr, beta1, beta2, epsilon)

    def c_adam_cuda(params, m_estimates, v_estimates, num_params, time_step, lr, beta1, beta2, epsilon):
        in_param_ptrs = (ctypes.POINTER(CTensor) * num_params)(*params)
        m_estimates_ptrs = (ctypes.POINTER(CTensor) * num_params)(*m_estimates)
        v_estimates_ptrs = (ctypes.POINTER(CTensor) * num_params)(*v_estimates)
        tensor_lib.adam_cuda(in_param_ptrs, m_estimates_ptrs, v_estimates_ptrs, num_params, time_step, lr, beta1, beta2, epsilon)

    def c_zero_grad_cpu(params, num_params):
        in_param_ptrs = (ctypes.POINTER(CTensor) * num_params)(*params)
        tensor_lib.zero_grad_cpu(in_param_ptrs, num_params)
    
    def c_zero_grad_cuda(params, num_params):
        in_param_ptrs = (ctypes.POINTER(CTensor) * num_params)(*params)
        tensor_lib.zero_grad_cuda(in_param_ptrs, num_params)

    # Device management
    def c_count_devices():
        return tensor_lib.count_devices()

    def c_is_cuda_available():
        return tensor_lib.is_cuda_available()

    def c_print_device_props():
        tensor_lib.print_device_props()

    def c_print_cuda_device_info(index):
        tensor_lib.print_cuda_device_info(index)

    def c_get_cuda_memory_info(device_id):
        c_str = tensor_lib.get_cuda_memory_info(device_id)
        if c_str:
            py_str = c_str.decode('utf-8')
            tensor_lib.free_char_ptr(c_str)
            return py_str
        return None

    # Register operations
    _register_op("add_grad", c_add_grad_op_cpu, c_add_grad_op_cuda)
    _register_op("sub_grad", c_sub_grad_op_cpu, c_sub_grad_op_cuda)
    _register_op("rsub_grad", c_rsub_grad_op_cpu, c_rsub_grad_op_cuda)
    _register_op("mul_grad", c_mul_grad_op_cpu, c_mul_grad_op_cuda)
    _register_op("pow_grad", c_pow_grad_op_cpu, c_pow_grad_op_cuda)
    _register_op("div_grad", c_div_grad_op_cpu, c_div_grad_op_cuda)
    _register_op("rdiv_grad", c_rdiv_grad_op_cpu, c_rdiv_grad_op_cuda)
    _register_op("matmul_grad", c_matmul_grad_op_cpu, c_matmul_grad_op_cuda)
    _register_op("conv_grad", c_conv_grad_op_cpu, c_conv_grad_op_cuda)
    _register_op("dot_grad", c_dot_grad_op_cpu, c_dot_grad_op_cuda)
    _register_op("relu_grad", c_relu_grad_op_cpu, c_relu_grad_op_cuda)
    _register_op("log_grad", c_log_grad_op_cpu, c_log_grad_op_cuda)
    _register_op("exp_grad", c_exp_grad_op_cpu, c_exp_grad_op_cuda)
    _register_op("softmax_grad", c_softmax_grad_op_cpu, c_softmax_grad_op_cuda)
    _register_op("abs_grad", c_abs_grad_op_cpu, c_abs_grad_op_cuda)
    _register_op("neg_grad", c_neg_grad_op_cpu, c_neg_grad_op_cuda)
    _register_op("clip_grad", c_clip_grad_op_cpu, c_clip_grad_op_cuda)
    _register_op("sum_grad", c_sum_grad_op_cpu, c_sum_grad_op_cuda)
    _register_op("mean_grad", c_mean_grad_op_cpu, c_mean_grad_op_cuda)
    _register_op("max_grad", c_max_grad_op_cpu, c_max_grad_op_cuda)
    _register_op("sum_full_grad", c_sum_full_grad_op_cpu, c_sum_full_grad_op_cuda)
    _register_op("mean_full_grad", c_mean_full_grad_op_cpu, c_mean_full_grad_op_cuda)
    _register_op("max_full_grad", c_max_full_grad_op_cpu, c_max_full_grad_op_cuda)
    _register_op("stack_grad", c_stack_grad_op_cpu, c_stack_grad_op_cuda)
    _register_op("concat_grad", c_concat_grad_op_cpu, c_concat_grad_op_cuda)

    _register_op("relu", c_relu_cpu, c_relu_cuda)
    _register_op("log", c_log_cpu, c_log_cuda)
    _register_op("exp", c_exp_cpu, c_exp_cuda)
    _register_op("softmax", c_softmax_cpu, c_softmax_cuda)
    _register_op("abs", c_abs_cpu, c_abs_cuda)
    _register_op("neg", c_neg_cpu, c_neg_cuda)
    _register_op("clip", c_clip_cpu, c_clip_cuda)

    _register_op("sum", c_sum_cpu, c_sum_cuda)
    _register_op("mean", c_mean_cpu, c_mean_cuda)
    _register_op("max", c_max_cpu, c_max_cuda)
    _register_op("sum_full", c_sum_full_cpu, c_sum_full_cuda)
    _register_op("mean_full", c_mean_full_cpu, c_mean_full_cuda)
    _register_op("max_full", c_max_full_cpu, c_max_full_cuda)

    _register_op("add", c_add_cpu, c_add_cuda)
    _register_op("sub", c_sub_cpu, c_sub_cuda)
    _register_op("mul", c_mul_cpu, c_mul_cuda)
    _register_op("div", c_div_cpu, c_div_cuda)
    _register_op("matmul", c_matmul_cpu, c_matmul_cuda)
    _register_op("conv", c_conv_cpu, c_conv_cuda)
    _register_op("dot", c_dot_cpu, c_dot_cuda)

    _register_op("add_scalar", c_add_scalar_cpu, c_add_scalar_cuda)
    _register_op("sub_scalar", c_sub_scalar_cpu, c_sub_scalar_cuda)
    _register_op("rsub_scalar", c_rsub_scalar_cpu, c_rsub_scalar_cuda)
    _register_op("mul_scalar", c_mul_scalar_cpu, c_mul_scalar_cuda)
    _register_op("div_scalar", c_div_scalar_cpu, c_div_scalar_cuda)
    _register_op("rdiv_scalar", c_rdiv_scalar_cpu, c_rdiv_scalar_cuda)
    _register_op("pow_scalar", c_pow_scalar_cpu, c_pow_scalar_cuda)
    _register_op("pow", c_pow_cpu, c_pow_cuda)

    _register_op("view", c_view, None)
    _register_op("unsqueeze", c_unsqueeze, None)
    _register_op("squeeze", c_squeeze, None)
    _register_op("transpose", c_transpose, None)
    _register_op("expand", c_expand, None)
    _register_op("broadcast", c_broadcast, None)
    _register_op("concat", c_concat_cpu, c_concat_cuda)

    _register_op("sgd", c_sgd_cpu, c_sgd_cuda)
    _register_op("zero_grad", c_zero_grad_cpu, c_zero_grad_cuda)
    _register_op("adam", c_adam_cpu, c_adam_cuda)
