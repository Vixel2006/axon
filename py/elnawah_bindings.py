import ctypes
import os
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
library_path = os.path.join(project_root, "build", "libelnawah.so")

tensor_lib = None
try:
    tensor_lib = ctypes.CDLL(library_path)
except OSError as e:
    print(f"Error loading shared library: {e}")
    print(f"Please ensure '{library_path}' exists and is accessible.")


# Define the C Tensor struct
class CTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("ndim", ctypes.c_int),
        ("shape", ctypes.POINTER(ctypes.c_int)),
        ("strides", ctypes.POINTER(ctypes.c_int)),
        ("requires_grad", ctypes.c_bool),
        ("grad", ctypes.POINTER(ctypes.c_float)),
        ("ctx", ctypes.c_void_p),
        ("owned_data", ctypes.c_bool),
        ("owned_grad", ctypes.c_bool),
    ]


if tensor_lib:
    # Define the C function signatures
    tensor_lib.numel.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    tensor_lib.numel.restype = ctypes.c_int

    tensor_lib.compute_strides.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    tensor_lib.compute_strides.restype = ctypes.POINTER(ctypes.c_int)

    tensor_lib.malloc_tensor_empty.argtypes = []
    tensor_lib.malloc_tensor_empty.restype = ctypes.POINTER(CTensor)

    tensor_lib.malloc_tensor_shape.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_bool,
    ]
    tensor_lib.malloc_tensor_shape.restype = ctypes.POINTER(CTensor)

    tensor_lib.malloc_tensor_full.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_bool,
        ctypes.POINTER(ctypes.c_float),
    ]
    tensor_lib.malloc_tensor_full.restype = ctypes.POINTER(CTensor)

    tensor_lib.free_tensor.argtypes = [ctypes.POINTER(CTensor)]
    tensor_lib.free_tensor.restype = None

    # Unary operations
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

    # Binary operations
    tensor_lib.add_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
    ]
    tensor_lib.add_op.restype = None

    tensor_lib.sub_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
    ]
    tensor_lib.sub_op.restype = None

    tensor_lib.mul_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
    ]
    tensor_lib.mul_op.restype = None

    tensor_lib.div_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
    ]
    tensor_lib.div_op.restype = None

    tensor_lib.matmul_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    tensor_lib.matmul_op.restype = None

    # Binary operations with scalars
    tensor_lib.add_scalar_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_float,
        ctypes.POINTER(CTensor),
    ]
    tensor_lib.add_scalar_op.restype = None

    tensor_lib.sub_scalar_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_float,
        ctypes.POINTER(CTensor),
    ]
    tensor_lib.sub_scalar_op.restype = None

    tensor_lib.rsub_scalar_op.argtypes = [
        ctypes.c_float,
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
    ]
    tensor_lib.rsub_scalar_op.restype = None

    tensor_lib.mul_scalar_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_float,
        ctypes.POINTER(CTensor),
    ]
    tensor_lib.mul_scalar_op.restype = None

    tensor_lib.div_scalar_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_float,
        ctypes.POINTER(CTensor),
    ]
    tensor_lib.div_scalar_op.restype = None

    tensor_lib.rdiv_scalar_op.argtypes = [
        ctypes.c_float,
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
    ]
    tensor_lib.rdiv_scalar_op.restype = None

    # Movement operations
    tensor_lib.view_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    tensor_lib.view_op.restype = None

    tensor_lib.squeeze_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
        ctypes.c_int,
    ]
    tensor_lib.squeeze_op.restype = None

    tensor_lib.unsqueeze_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
        ctypes.c_int,
    ]
    tensor_lib.unsqueeze_op.restype = None

    tensor_lib.transpose_op.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor),
        ctypes.c_int,
        ctypes.c_int,
    ]
    tensor_lib.transpose_op.restype = None

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

    def tensor_to_numpy(tensor_ptr):
        if not tensor_ptr:
            return None

        tensor = tensor_ptr.contents

        if tensor.ndim == 0:
            return np.array(tensor.data[0])

        shape = [tensor.shape[i] for i in range(tensor.ndim)]
        size = np.prod(shape)

        data = [tensor.data[i] for i in range(size)]

        return np.array(data, dtype=np.float32).reshape(shape)

    def print_tensor_info(tensor_ptr):
        if not tensor_ptr:
            print("NULL tensor")
            return

        tensor = tensor_ptr.contents
        print(f"Tensor info:")
        print(f"  ndim: {tensor.ndim}")

        if tensor.ndim > 0:
            shape = [tensor.shape[i] for i in range(tensor.ndim)]
            strides = [tensor.strides[i] for i in range(tensor.ndim)]
            print(f"  shape: {shape}")
            print(f"  strides: {strides}")

        print(f"  requires_grad: {tensor.requires_grad}")

        # Print some data values
        if tensor.ndim == 0:
            print(f"  data: {tensor.data[0]}")
            if tensor.grad:
                print(f"  grad: {tensor.grad[0]}")
        else:
            size = np.prod([tensor.shape[i] for i in range(tensor.ndim)])
            data_sample = [tensor.data[i] for i in range(min(5, size))]
            print(f"  data (first 5): {data_sample}")
            if tensor.grad:
                grad_sample = [tensor.grad[i] for i in range(min(5, size))]
                print(f"  grad (first 5): {grad_sample}")

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

    def tensor_to_numpy(tensor_ptr):
        print("C backend not available: tensor_to_numpy()")
        return None

    def print_tensor_info(tensor_ptr):
        print("C backend not available: print_tensor_info()")
