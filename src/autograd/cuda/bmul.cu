#include "autograd/ops.h"
#include "tensor.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

// This is a reasonable maximum, consistent with many frameworks.
#define MAX_DIMS 8

// Kernel for non-broadcasted, element-wise multiplication backward pass.
// This is faster when no broadcasting is involved as it avoids atomic operations.
__global__ void mul_tensor_backward_kernel(const float* out_grad_p,
                                         const float* a_data_p,
                                         const float* b_data_p,
                                         float* a_grad_p,
                                         float* b_grad_p,
                                         bool a_req_grad,
                                         bool b_req_grad,
                                         size_t num_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (size_t i = index; i < num_elements; i += stride) {
        const float grad_out = out_grad_p[i];
        if (a_req_grad) {
            a_grad_p[i] += grad_out * b_data_p[i];
        }
        if (b_req_grad) {
            b_grad_p[i] += grad_out * a_data_p[i];
        }
    }
}

// Broadcasting-aware kernel for multiplication backward pass.
// It uses the same stride-based indexing as your forward pass kernel.
__global__ void mul_broadcast_backward_kernel(const float* out_grad_p,
                                              const float* a_data_p,
                                              const int64_t* a_strides_b, // Broadcasted strides
                                              const float* b_data_p,
                                              const int64_t* b_strides_b, // Broadcasted strides
                                              float* a_grad_p,
                                              float* b_grad_p,
                                              bool a_req_grad,
                                              bool b_req_grad,
                                              const int64_t* out_shape,
                                              int out_dims,
                                              size_t num_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) {
        return;
    }

    // This logic mirrors your forward pass to find the correct source elements.
    int64_t temp_i = i;
    size_t a_offset = 0;
    size_t b_offset = 0;

    for (int d = out_dims - 1; d >= 0; --d) {
        int64_t coord = temp_i % out_shape[d];
        temp_i /= out_shape[d];
        a_offset += coord * a_strides_b[d];
        b_offset += coord * b_strides_b[d];
    }

    const float grad_out = out_grad_p[i];

    if (a_req_grad) {
        // atomicAdd is crucial here. It performs the reduction sum for gradients
        // across broadcasted dimensions, preventing race conditions.
        atomicAdd(&a_grad_p[a_offset], grad_out * b_data_p[b_offset]);
    }
    if (b_req_grad) {
        atomicAdd(&b_grad_p[b_offset], grad_out * a_data_p[a_offset]);
    }
}


// Backward pass for tensor-scalar multiplication.
// WARNING: This kernel re-calculates the scalar by division (out / a).
// This is numerically unstable if a_data can be zero. A better design
// would be to save the scalar value in the context during the forward pass.
__global__ void mul_scalar_backward_kernel(const float* out_grad_p,
                                           const float* a_data_p,
                                           const float* out_data_p,
                                           float* a_grad_p,
                                           size_t num_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (size_t i = index; i < num_elements; i += stride) {
        const float a_val = a_data_p[i];
        if (a_val != 0.0f) {
            const float scalar_operand = out_data_p[i] / a_val;
            a_grad_p[i] += out_grad_p[i] * scalar_operand;
        }
    }
}

// Helper to copy shape/stride data to the GPU
template<typename T>
void copy_to_device(const std::vector<T>& vec, T*& d_ptr) {
    if (vec.empty()) {
        d_ptr = nullptr;
        return;
    }
    size_t size_in_bytes = vec.size() * sizeof(T);
    CUDA_CHECK(cudaMalloc(&d_ptr, size_in_bytes));
    CUDA_CHECK(cudaMemcpy(d_ptr, vec.data(), size_in_bytes, cudaMemcpyHostToDevice));
}


void CudaAutograd::mul(Tensor& out, std::vector<Tensor>& prev) {
    const size_t num_elements = out.numel();
    if (num_elements == 0) return;

    const float* out_grad_p = static_cast<const float*>(out.grad_ptr().get());
    if (!out_grad_p) {
        throw std::runtime_error("Output gradient pointer is null in 'mul' backward pass (CUDA).");
    }

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    if (prev.size() == 2) { // Tensor-Tensor multiplication
        Tensor& a = prev[0];
        Tensor& b = prev[1];
        const bool a_req_grad = a.requires_grad();
        const bool b_req_grad = b.requires_grad();

        if (!a_req_grad && !b_req_grad) return;

        const float* a_data_p = static_cast<const float*>(a.data_ptr().get());
        const float* b_data_p = static_cast<const float*>(b.data_ptr().get());
        float* a_grad_p = a_req_grad ? static_cast<float*>(a.grad_ptr().get()) : nullptr;
        float* b_grad_p = b_req_grad ? static_cast<float*>(b.grad_ptr().get()) : nullptr;
        
        // Check if broadcasting occurred in the forward pass.
        bool is_broadcasted = a.shape() != out.shape() || b.shape() != out.shape();

        if (!is_broadcasted) {
            // Use the faster, non-atomic kernel if shapes match.
            mul_tensor_backward_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                out_grad_p, a_data_p, b_data_p, a_grad_p, b_grad_p, a_req_grad, b_req_grad, num_elements
            );
        } else {
            // Broadcasting path
            if (out.ndim() > MAX_DIMS) {
                throw std::runtime_error("Tensor exceeds maximum supported dimensions for broadcasting.");
            }

            // Get broadcasted views to easily access the correct strides, just like in the forward pass.
            Tensor a_broad = a.broadcast(out.shape());
            Tensor b_broad = b.broadcast(out.shape());

            // Allocate and copy shape/stride metadata to the GPU.
            int64_t *d_out_shape, *d_a_strides, *d_b_strides;
            copy_to_device(out.shape(), d_out_shape);
            copy_to_device(a_broad.strides(), d_a_strides);
            copy_to_device(b_broad.strides(), d_b_strides);

            mul_broadcast_backward_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                out_grad_p, a_data_p, d_a_strides, b_data_p, d_b_strides,
                a_grad_p, b_grad_p, a_req_grad, b_req_grad,
                d_out_shape, out.ndim(), num_elements
            );

            // Free the temporary metadata from the GPU.
            CUDA_CHECK(cudaFree(d_out_shape));
            CUDA_CHECK(cudaFree(d_a_strides));
            CUDA_CHECK(cudaFree(d_b_strides));
        }

    } else if (prev.size() == 1) { // Tensor-Scalar multiplication
        Tensor& a = prev[0];
        if (!a.requires_grad()) return;

        const float* a_data_p = static_cast<const float*>(a.data_ptr().get());
        const float* out_data_p = static_cast<const float*>(out.data_ptr().get());
        float* a_grad_p = static_cast<float*>(a.grad_ptr().get());

        mul_scalar_backward_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            out_grad_p, a_data_p, out_data_p, a_grad_p, num_elements
        );
    } else {
        throw std::runtime_error("Invalid number of inputs for mul backward pass (CUDA).");
    }

    CUDA_CHECK(cudaGetLastError());
}
