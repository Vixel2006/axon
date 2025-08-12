#include "autograd/ops.h"
#include "tensor.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void bexp_kernel(const float* out_grad_p,
                            const float* out_p,
                            float* a_grad_p,
                            size_t num_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (size_t i = index; i < num_elements; i += stride) {
        const float grad = out_grad_p[i];
        const float out_val = out_p[i];
        
        a_grad_p[i] += grad * out_val;
    }
}

void CudaAutograd::exp(Tensor& out, std::vector<Tensor>& prev) {
    if (prev.size() != 1) {
        throw std::runtime_error("Exp backward operation requires exactly one previous tensor.");
    }

    Tensor t = out;
    Tensor& a = prev[0];
    
    if (!a.requires_grad()) {
        return;
    }

    const size_t num_elements = a.numel();
    if (num_elements == 0) {
        return;
    }

    const float* out_grad_p = static_cast<const float*>(t.grad_ptr().get());
    const float* out_p = static_cast<const float*>(t.data_ptr().get());
    float* a_grad_p = static_cast<float*>(a.grad_ptr().get());

    if (!out_grad_p || !out_p || !a_grad_p) {
        throw std::runtime_error("A data or gradient pointer is null in 'exp' backward pass (CUDA).");
    }

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    bexp_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        out_grad_p,
        out_p,
        a_grad_p,
        num_elements
    );

    CUDA_CHECK(cudaGetLastError());
}
