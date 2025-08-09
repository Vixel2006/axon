#include "autograd/ops.h"
#include "tensor.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <stdexcept>

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

void CudaAutograd::mul(const Tensor& out, std::vector<Tensor>& prev) {
    Tensor t = out;
    const size_t num_elements = out.numel();
    if (num_elements == 0) return;

    const float* out_grad_p = static_cast<const float*>(t.grad_ptr().get());
    if (!out_grad_p) {
        throw std::runtime_error("Output gradient pointer is null in 'mul' backward pass (CUDA).");
    }

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    if (prev.size() == 2) {
        Tensor& a = prev[0];
        Tensor& b = prev[1];
        const bool a_req_grad = a.requires_grad();
        const bool b_req_grad = b.requires_grad();

        if (!a_req_grad && !b_req_grad) return;

        const float* a_data_p = static_cast<const float*>(a.data_ptr().get());
        const float* b_data_p = static_cast<const float*>(b.data_ptr().get());
        float* a_grad_p = a_req_grad ? static_cast<float*>(a.grad_ptr().get()) : nullptr;
        float* b_grad_p = b_req_grad ? static_cast<float*>(b.grad_ptr().get()) : nullptr;
        
        mul_tensor_backward_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            out_grad_p, a_data_p, b_data_p, a_grad_p, b_grad_p, a_req_grad, b_req_grad, num_elements
        );
    }
    else if (prev.size() == 1) {
        Tensor& a = prev[0];
        if (!a.requires_grad()) return;

        const float* a_data_p = static_cast<const float*>(a.data_ptr().get());
        const float* out_data_p = static_cast<const float*>(out.data_ptr().get());
        float* a_grad_p = static_cast<float*>(a.grad_ptr().get());

        mul_scalar_backward_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            out_grad_p, a_data_p, out_data_p, a_grad_p, num_elements
        );
    }
    else {
        throw std::runtime_error("Invalid number of inputs for mul backward pass (CUDA).");
    }

    CUDA_CHECK(cudaGetLastError());
}
