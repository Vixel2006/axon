import ctypes
from os import wait
import numpy as np
from .c_library_loader import tensor_lib
from .c_function_signatures import *
from .ctypes_definitions import CTensor, CNode, CSharedPtr, CDtype, CDevice


if tensor_lib:

    # Python wrapper functions
    def c_numel(shape, ndim):
        if ndim == 0:
            return 1
        c_shape = (ctypes.c_int * ndim)(*shape)
        return tensor_lib.numel(c_shape, ndim)

    def c_set_ones_grad(tensor_ptr):
        return tensor_lib.set_ones_grad(tensor_ptr)

    def c_compute_strides(shape, ndim):
        if ndim == 0:
            return None
        c_shape = (ctypes.c_int * ndim)(*shape)
        return tensor_lib.compute_strides(c_shape, ndim)

    

    def c_malloc_tensor_shape(shape, ndim, requires_grad, dtype, device):
        c_shape = (ctypes.c_int * ndim)(*shape)
        return tensor_lib.tmalloc_shape(c_shape, ndim, dtype, device, requires_grad)

    def c_malloc_tensor_full(shape, ndim, strides, dtype, device, data, requires_grad, grad=None):
        if not shape or ndim <= 0:
            raise ValueError("Invalid shape or ndim")

        c_shape = (ctypes.c_int * ndim)(*shape)

        c_strides = None
        if strides is not None:
            c_strides = strides

        if isinstance(data, np.ndarray):
            data = data.flatten().astype(np.float32).tolist()
        elif isinstance(data, (list, tuple)):
            data = [float(x) for x in data]
        else:
            raise ValueError("Data must be a list, tuple, or numpy array")

        expected_size = 1
        for dim in shape:
            expected_size *= dim

        if len(data) != expected_size:
            raise ValueError(
                f"Data size ({len(data)}) doesn't match expected size ({expected_size})"
            )

        c_data_array = (ctypes.c_float * len(data))(*data)
        c_data_shared_ptr = tensor_lib.palloc(ctypes.cast(c_data_array, ctypes.c_void_p), len(data), dtype, device).contents

        c_grad_shared_ptr = CSharedPtr() # Initialize with default values
        if requires_grad and grad is None:
            # If requires_grad is True but no initial grad is provided, create a zeroed grad
            if ndim == 0:
                zero_grad_val = 0.0
                c_grad_array = (ctypes.c_float * 1)(zero_grad_val)
                c_grad_shared_ptr = tensor_lib.palloc(ctypes.cast(c_grad_array, ctypes.c_void_p), 1, dtype, device).contents
            else:
                # Determine size based on data length, assuming grad should match data size
                grad_size = len(data) if isinstance(data, (list, tuple, np.ndarray)) else 1
                c_grad_array = (ctypes.c_float * grad_size)(*[0.0] * grad_size)
                c_grad_shared_ptr = tensor_lib.palloc(ctypes.cast(c_grad_array, ctypes.c_void_p), grad_size, dtype, device).contents
        elif grad is not None:
            if ndim == 0:
                if isinstance(grad, (list, tuple, np.ndarray)):
                    grad_val = float(grad[0]) if len(grad) > 0 else 0.0
                else:
                    grad_val = float(grad)
                c_grad_array = (ctypes.c_float * 1)(grad_val)
                c_grad_shared_ptr = tensor_lib.palloc(c_grad_array, 1, dtype, device).contents
            else:
                if isinstance(grad, np.ndarray):
                    grad = grad.flatten().astype(np.float32).tolist()
                elif isinstance(grad, (list, tuple)):
                    grad = [float(x) for x in grad]
                else:
                    raise ValueError("Grad must be a list, tuple, or numpy array")

                expected_size = 1
                for dim in shape:
                    expected_size *= dim

                if len(grad) != expected_size:
                    raise ValueError("Gradient size must match data size")

                c_grad_array = (ctypes.c_float * len(grad))(*grad)
                c_grad_shared_ptr = tensor_lib.palloc(ctypes.cast(c_grad_array, ctypes.c_void_p), len(grad), dtype, device).contents

        c_tensor_ptr = tensor_lib.tmalloc_full(
                c_shape, ndim, c_strides, dtype, device, c_data_shared_ptr, requires_grad, c_grad_shared_ptr
            )
        return c_tensor_ptr

    def c_free_tensor(tensor_ptr):
        if tensor_ptr and tensor_ptr.contents:
            # Free the shared pointers first
            if tensor_ptr.contents.data:
                tensor_lib.pfree(tensor_ptr.contents.data)
            if tensor_ptr.contents.grad:
                tensor_lib.pfree(tensor_ptr.contents.grad)
            tensor_lib.tfree(ctypes.byref(tensor_ptr))

    def c_zeros(shape, ndim, requires_grad, dtype, device):
        c_shape = (ctypes.c_int * ndim)(*shape)
        return tensor_lib.zeros(c_shape, ndim, requires_grad, dtype, device)

    def c_ones(shape, ndim, requires_grad, dtype, device):
        if ndim == 0:
            return tensor_lib.ones(None, 0, requires_grad, dtype, device)
        c_shape = (ctypes.c_int * ndim)(*shape)
        return tensor_lib.ones(c_shape, ndim, requires_grad, dtype, device)

    def c_randn(shape, ndim, dtype, device, seed, requires_grad):
        if ndim == 0:
            return tensor_lib.randn(None, 0, dtype.value, device.value, seed, requires_grad)
        c_shape = (ctypes.c_int * ndim)(*shape)
        return tensor_lib.randn(c_shape, ndim, dtype.value, device.value, seed, requires_grad)

    def c_uniform(shape, ndim, dtype, device, low, high, requires_grad):
        if ndim == 0:
            return tensor_lib.uniform(None, 0, dtype.value, device.value, low, high, requires_grad)
        c_shape = (ctypes.c_int * ndim)(*shape)
        return tensor_lib.uniform(c_shape, ndim, dtype.value, device.value, low, high, requires_grad)

    def c_malloc_node(
        out_tensor_ptr, prev_tensor_ptrs, n_prev, extras, forward_fn, backward_fn
    ):
        c_prev_array = (ctypes.POINTER(CTensor) * n_prev)(*prev_tensor_ptrs)
        return tensor_lib.nmalloc(
            out_tensor_ptr, c_prev_array, n_prev, extras, forward_fn, backward_fn
        )

    def c_free_node(node_ptr):
        if node_ptr:
            tensor_lib.nfree(node_ptr)

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
        print(extras)
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

    def c_broadcast(in_tensor_ptr, out_tensor_ptr, ndim, shape):
        c_shape = (ctypes.c_int * ndim)(*shape)
        tensor_lib.broadcast_op(in_tensor_ptr, out_tensor_ptr, ndim, c_shape)

    def c_concat(in_tensors, out_tensor_ptr, num_tensors, axis):
        in_tensor_ptrs = (ctypes.POINTER(CTensor) * num_tensors)(*in_tensors)
        tensor_lib.concat_op(in_tensor_ptrs, out_tensor_ptr, num_tensors, axis)

    def c_stack(in_tensors, out_tensor_ptr, num_tensors, axis):
        in_tensor_ptrs = (ctypes.POINTER(CTensor) * num_tensors)(*in_tensors)
        tensor_lib.stack_op(in_tensor_ptrs, out_tensor_ptr, num_tensors, axis)

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

    def c_numel(shape, ndim):
        print("C backend not available: numel()")
        return 0

    def c_set_debug_mode(enable):
        print("C backend not available: set_debug_mode()")

    def c_compute_strides(shape, ndim):
        print("C backend not available: compute_strides()")
        return None

    

    def c_tmalloc_shape(shape, ndim, requires_grad):
        print("C backend not available: tmalloc_shape()")
        return None

    def c_tmalloc_full(shape, ndim, strides, data, requires_grad, grad=None):
        print("C backend not available: tmalloc_full()")
        return None

    def c_zeros(shape, ndim, requires_grad):
        print("C backend not available: zeros()")
        return None

    def c_ones(shape, ndim, requires_grad):
        print("C backend not available: ones()")
        return None

    def c_randn(shape, ndim, seed, requires_grad):
        print("C backend not available: randn()")
        return None

    def c_uniform(shape, ndim, low, high, requires_grad):
        print("C backend not available: uniform()")
        return None

    def c_free_tensor(tensor_ptr):
        print("C backend not available: free_tensor()")

    def c_relu(in_tensor_ptr, out_tensor_ptr):
        print("C backend not available: relu()")

    def c_log(in_tensor_ptr, out_tensor_ptr):
        print("C backend not available: log()")

    def c_exp(in_tensor_ptr, out_tensor_ptr):
        print("C backend not available: exp()")

    def c_softmax(in_tensor_ptr, out_tensor_ptr):
        print("C backend not available: softmax()")

    def c_abs(in_tensor_ptr, out_tensor_ptr):
        print("C backend not available: abs()")

    def c_neg(in_tensor_ptr, out_tensor_ptr):
        print("C backend not available: neg()")

    def c_add(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr):
        print("C backend not available: add()")

    def c_sub(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr):
        print("C backend not available: sub()")

    def c_mul(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr):
        print("C backend not available: mul()")

    def c_div(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr):
        print("C backend not available: div()")

    def c_matmul(a_tensor_ptr, b_tensor_ptr, out_tensor_ptr, N, K, M):
        print("C backend not available: matmul()")

    def c_conv(
        a_tensor_ptr, b_tensor_ptr, out_tensor_ptr, kernel_size, stride, padding
    ):
        print("C backend not available: conv2d()")

    def c_broadcast(in_tensor_ptr, out_tensor_ptr, shape, ndim):
        print("C backend not available: broadcast()")
