import numpy as np
from .ctypes_definitions import CTensor

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
            print(f"  grad (first 5): {grad_sample})")
