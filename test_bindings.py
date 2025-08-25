from py.elnawah_bindings import c_numel, c_malloc_tensor_shape, c_free_tensor, CTensor
import ctypes

# Test c_numel
shape = [2, 3, 4]
ndim = len(shape)
num_elements = c_numel(shape, ndim)
print(f"Number of elements for shape {shape}: {num_elements}")

# Test c_malloc_tensor_shape
requires_grad = True
tensor_ptr = c_malloc_tensor_shape(shape, ndim, requires_grad)

if tensor_ptr:
    print(f"Tensor allocated at: {tensor_ptr}")
    # Accessing members of the allocated Tensor
    tensor = tensor_ptr.contents
    print(f"Tensor ndim: {tensor.ndim}")
    print(f"Tensor requires_grad: {tensor.requires_grad}")

    # Accessing shape (requires iterating through the pointer)
    print("Tensor shape:", end=" ")
    for i in range(tensor.ndim):
        print(tensor.shape[i], end=" ")
    print()

    # Accessing strides (requires iterating through the pointer)
    print("Tensor strides:", end=" ")
    for i in range(tensor.ndim):
        print(tensor.strides[i], end=" ")
    print()

    # Free the tensor
    c_free_tensor(tensor_ptr)
    print("Tensor freed.")
else:
    print("Failed to allocate tensor.")
