import ctypes
import numpy as np
from .c_library_loader import tensor_lib
from .ctypes_definitions import CTensor, CNode, BackwardFnType

if tensor_lib:
    tensor_lib.malloc_node.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.POINTER(CTensor)),
        ctypes.c_int,
        ctypes.c_void_p,
        BackwardFnType,
    ]
    tensor_lib.malloc_node.restype = CNode

    tensor_lib.free_node.argtypes = [ctypes.POINTER(CNode)]
    tensor_lib.free_node.restype = None

    tensor_lib.malloc_tensor_empty.restype = ctypes.POINTER(CTensor)
    tensor_lib.malloc_tensor_shape.restype = ctypes.POINTER(CTensor)
    tensor_lib.malloc_tensor_full.restype = ctypes.POINTER(CTensor)
    tensor_lib.free_tensor.argtypes = [ctypes.POINTER(CTensor)]
    tensor_lib.free_tensor.restype = None

    tensor_lib.numel.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    tensor_lib.numel.restype = ctypes.c_int

    tensor_lib.compute_strides.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    tensor_lib.compute_strides.restype = ctypes.POINTER(ctypes.c_int)

    # Set argtypes and restypes for other functions as needed
    # Unary ops
    tensor_lib.relu_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    tensor_lib.relu_op.restype = None
    tensor_lib.log_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    tensor_lib.log_op.restype = None
    tensor_lib.exp_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    tensor_lib.exp_op.restype = None
    tensor_lib.softmax_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    tensor_lib.softmax_op.restype = None
    tensor_lib.abs_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    tensor_lib.abs_op.restype = None
    tensor_lib.neg_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    tensor_lib.neg_op.restype = None

    # Reduction ops
    tensor_lib.sum_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int, ctypes.c_bool]
    tensor_lib.sum_op.restype = None
    tensor_lib.mean_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int, ctypes.c_bool]
    tensor_lib.mean_op.restype = None
    tensor_lib.max_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int, ctypes.c_bool]
    tensor_lib.max_op.restype = None

    # Binary ops
    tensor_lib.add_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    tensor_lib.add_op.restype = None
    tensor_lib.sub_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    tensor_lib.sub_op.restype = None
    tensor_lib.mul_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    tensor_lib.mul_op.restype = None
    tensor_lib.div_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    tensor_lib.div_op.restype = None
    tensor_lib.matmul_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int, ctypes.c_int, ctypes.c_int]
    tensor_lib.matmul_op.restype = None

    # Binary ops with scalars
    tensor_lib.add_scalar_op.argtypes = [ctypes.POINTER(CTensor), ctypes.c_float, ctypes.POINTER(CTensor)]
    tensor_lib.add_scalar_op.restype = None
    tensor_lib.sub_scalar_op.argtypes = [ctypes.POINTER(CTensor), ctypes.c_float, ctypes.POINTER(CTensor)]
    tensor_lib.sub_scalar_op.restype = None
    tensor_lib.rsub_scalar_op.argtypes = [ctypes.c_float, ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    tensor_lib.rsub_scalar_op.restype = None
    tensor_lib.mul_scalar_op.argtypes = [ctypes.POINTER(CTensor), ctypes.c_float, ctypes.POINTER(CTensor)]
    tensor_lib.mul_scalar_op.restype = None
    tensor_lib.div_scalar_op.argtypes = [ctypes.POINTER(CTensor), ctypes.c_float, ctypes.POINTER(CTensor)]
    tensor_lib.div_scalar_op.restype = None
    tensor_lib.rdiv_scalar_op.argtypes = [ctypes.c_float, ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    tensor_lib.rdiv_scalar_op.restype = None

    # Movement ops
    tensor_lib.view_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    tensor_lib.view_op.restype = None
    tensor_lib.unsqueeze_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int]
    tensor_lib.unsqueeze_op.restype = None
    tensor_lib.squeeze_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int]
    tensor_lib.squeeze_op.restype = None
    tensor_lib.transpose_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int, ctypes.c_int]
    tensor_lib.transpose_op.restype = None
    tensor_lib.expand_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.c_int)]
    tensor_lib.expand_op.restype = None

    # Grad ops
    tensor_lib.add_grad_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.POINTER(CTensor)), ctypes.c_int, ctypes.c_void_p]
    tensor_lib.add_grad_op.restype = None
    tensor_lib.sub_grad_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.POINTER(CTensor)), ctypes.c_int, ctypes.c_void_p]
    tensor_lib.sub_grad_op.restype = None
    tensor_lib.rsub_grad_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.POINTER(CTensor)), ctypes.c_int, ctypes.c_void_p]
    tensor_lib.rsub_grad_op.restype = None
    tensor_lib.mul_grad_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.POINTER(CTensor)), ctypes.c_int, ctypes.c_void_p]
    tensor_lib.mul_grad_op.restype = None
    tensor_lib.div_grad_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.POINTER(CTensor)), ctypes.c_int, ctypes.c_void_p]
    tensor_lib.div_grad_op.restype = None
    tensor_lib.rdiv_grad_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.POINTER(CTensor)), ctypes.c_int, ctypes.c_void_p]
    tensor_lib.rdiv_grad_op.restype = None
    tensor_lib.relu_grad_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.POINTER(CTensor)), ctypes.c_int, ctypes.c_void_p]
    tensor_lib.relu_grad_op.restype = None
    tensor_lib.log_grad_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.POINTER(CTensor)), ctypes.c_int, ctypes.c_void_p]
    tensor_lib.log_grad_op.restype = None
    tensor_lib.exp_grad_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.POINTER(CTensor)), ctypes.c_int, ctypes.c_void_p]
    tensor_lib.exp_grad_op.restype = None
    tensor_lib.softmax_grad_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.POINTER(CTensor)), ctypes.c_int, ctypes.c_void_p]
    tensor_lib.softmax_grad_op.restype = None
    tensor_lib.abs_grad_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.POINTER(CTensor)), ctypes.c_int, ctypes.c_void_p]
    tensor_lib.abs_grad_op.restype = None
    tensor_lib.neg_grad_op.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.POINTER(CTensor)), ctypes.c_int, ctypes.c_void_p]
    tensor_lib.neg_grad_op.restype = None

    # Python wrapper functions
    # Python wrapper functions
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

    def c_malloc_tensor_empty():
        return tensor_lib.malloc_tensor_empty()

    def c_malloc_tensor_shape(shape, ndim, requires_grad):
        if ndim == 0:
            return tensor_lib.malloc_tensor_shape(None, 0, requires_grad)
        c_shape = (ctypes.c_int * ndim)(*shape)
        return tensor_lib.malloc_tensor_shape(c_shape, ndim, requires_grad)

    def c_malloc_tensor_full(shape, ndim, strides, data, requires_grad, grad=None):
        if ndim == 0:
            if isinstance(data, (list, tuple, np.ndarray)):
                data_val = float(data[0]) if len(data) > 0 else 0.0
            else:
                data_val = float(data)

            c_data = (ctypes.c_float * 1)(data_val)
            c_grad = None
            if grad is not None:
                if isinstance(grad, (list, tuple, np.ndarray)):
                    grad_val = float(grad[0]) if len(grad) > 0 else 0.0
                else:
                    grad_val = float(grad)
                c_grad = (ctypes.c_float * 1)(grad_val)

            return tensor_lib.malloc_tensor_full(
                None, 0, None, c_data, requires_grad, c_grad
            )

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

        c_data = (ctypes.c_float * len(data))(*data)

        c_grad = None
        if grad is not None:
            if isinstance(grad, np.ndarray):
                grad = grad.flatten().astype(np.float32).tolist()
            elif isinstance(grad, (list, tuple)):
                grad = [float(x) for x in grad]
            else:
                raise ValueError("Grad must be a list, tuple, or numpy array")

            if len(grad) != len(data):
                raise ValueError("Gradient size must match data size")

            c_grad = (ctypes.c_float * len(grad))(*grad)

        return tensor_lib.malloc_tensor_full(
            c_shape, ndim, c_strides, c_data, requires_grad, c_grad
        )

    def c_free_tensor(tensor_ptr):
        if tensor_ptr:
            tensor_lib.free_tensor(tensor_ptr)

    def c_malloc_node(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras, backward_fn):
        c_prev_array = (ctypes.POINTER(CTensor) * n_prev)(*prev_tensor_ptrs)
        return tensor_lib.malloc_node(
            out_tensor_ptr, c_prev_array, n_prev, extras, backward_fn
        )

    def c_free_node(node_ptr):
        if node_ptr:
            tensor_lib.free_node(node_ptr)

    def c_add_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.add_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_sub_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.sub_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_rsub_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.rsub_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_mul_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.mul_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_div_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.div_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

    def c_rdiv_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras):
        tensor_lib.rdiv_grad_op(out_tensor_ptr, prev_tensor_ptrs, n_prev, extras)

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

    # Reduction operations wrappers
    def c_sum(in_tensor_ptr, out_tensor_ptr, axis, keepdim):
        tensor_lib.sum_op(in_tensor_ptr, out_tensor_ptr, axis, keepdim)

    def c_mean(in_tensor_ptr, out_tensor_ptr, axis, keepdim):
        tensor_lib.mean_op(in_tensor_ptr, out_tensor_ptr, axis, keepdim)

    def c_max(in_tensor_ptr, out_tensor_ptr, axis, keepdim):
        tensor_lib.max_op(in_tensor_ptr, out_tensor_ptr, axis, keepdim)

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

else:

    def c_numel(shape, ndim):
        print("C backend not available: numel()")
        return 0

    def c_compute_strides(shape, ndim):
        print("C backend not available: compute_strides()")
        return None

    def c_malloc_tensor_empty():
        print("C backend not available: malloc_tensor_empty()")
        return None

    def c_malloc_tensor_shape(shape, ndim, requires_grad):
        print("C backend not available: malloc_tensor_shape()")
        return None

    def c_malloc_tensor_full(shape, ndim, strides, data, requires_grad, grad=None):
        print("C backend not available: malloc_tensor_full()")
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
