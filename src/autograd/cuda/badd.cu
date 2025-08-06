#include "autograd/ops.h"
#include "tensor.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void badd_kernel(const float* out_grad_p,
                            float* a_grad_p,
                            float* b_grad_p,
                            bool a_req_grad,
                            bool b_req_grad,
                            size_t num_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (size_t i = index; i < num_elements; i += stride) {
        const float grad = out_grad_p[i];
        
        if (a_req_grad) {
            a_grad_p[i] += grad;
        }

        if (b_req_grad) {
            b_grad_p[i] += grad;
        }
    }
}

void CudaAutograd::add(const Tensor& out, std::vector<Tensor>& prev) {
    Tensor t = out;
    Tensor& a = prev[0];
    Tensor& b = prev[1];
    
    const bool a_req_grad = a.requires_grad();
    const bool b_req_grad = b.requires_grad();

    if (!a_req_grad && !b_req_grad) {
        return;
    }

    const size_t num_elements = a.numel();
    if (num_elements == 0) {
        return;
    }

    const float* out_grad_p = static_cast<const float*>(t.grad_ptr().get());
    float* a_grad_p = static_cast<float*>(a.grad_ptr().get());
    float* b_grad_p = static_cast<float*>(b.grad_ptr().get());

    if (!out_grad_p || !a_grad_p || !b_grad_p) {
        throw std::runtime_error("A gradient pointer is null in 'add' backward pass (CUDA).");
    }

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    badd_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        out_grad_p,
        a_grad_p,
        b_grad_p,
        a_req_grad,
        b_req_grad,
        num_elements
    );

    CUDA_CHECK(cudaGetLastError());
}
