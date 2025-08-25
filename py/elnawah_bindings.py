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
        """
        Create a tensor with full initialization.

        Args:
            shape: Shape of the tensor (list/tuple of ints)
            ndim: Number of dimensions
            strides: Strides array (list/tuple of ints, or None for default)
            data: Data array (list, numpy array, or array of floats)
            requires_grad: Whether gradient computation is required
            grad: Gradient array (optional, same format as data)
        """
        # Handle scalar case
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

        # Validate inputs
        if not shape or ndim <= 0:
            raise ValueError("Invalid shape or ndim")

        # Convert shape to C array
        c_shape = (ctypes.c_int * ndim)(*shape)

        # Handle strides
        c_strides = None
        if strides is not None:
            c_strides = strides

        # Convert data to proper format
        if isinstance(data, np.ndarray):
            data = data.flatten().astype(np.float32).tolist()
        elif isinstance(data, (list, tuple)):
            data = [float(x) for x in data]
        else:
            raise ValueError("Data must be a list, tuple, or numpy array")

        # Calculate expected size
        expected_size = 1
        for dim in shape:
            expected_size *= dim

        if len(data) != expected_size:
            raise ValueError(
                f"Data size ({len(data)}) doesn't match expected size ({expected_size})"
            )

        # Create C data array
        c_data = (ctypes.c_float * len(data))(*data)

        # Handle gradient
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

    def tensor_to_numpy(tensor_ptr):
        """Convert a C tensor to a numpy array"""
        if not tensor_ptr:
            return None

        tensor = tensor_ptr.contents

        # Handle scalar case
        if tensor.ndim == 0:
            return np.array(tensor.data[0])

        # Get shape
        shape = [tensor.shape[i] for i in range(tensor.ndim)]
        size = np.prod(shape)

        # Get data
        data = [tensor.data[i] for i in range(size)]

        # Reshape and return
        return np.array(data, dtype=np.float32).reshape(shape)

    def print_tensor_info(tensor_ptr):
        """Debug function to print tensor information"""
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
    # Provide dummy functions if the library couldn't be loaded
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

    def tensor_to_numpy(tensor_ptr):
        print("C backend not available: tensor_to_numpy()")
        return None

    def print_tensor_info(tensor_ptr):
        print("C backend not available: print_tensor_info()")
