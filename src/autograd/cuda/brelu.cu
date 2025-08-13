#include "autograd/ops.h"
#include "tensor.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void relu_kernel(const float* out_grad_p,
                             const float* parent_data_p,
                             float* parent_grad_p,
                             size_t num_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (size_t i = index; i < num_elements; i += stride) {
        if (parent_data_p[i] > 0) {
            atomicAdd(&parent_grad_p[i], out_grad_p[i]);
        }
    }
}

void CudaAutograd::relu(Tensor& out, std::vector<Tensor>& prev) {
    const size_t num_elements = out.numel();
    if (num_elements == 0) {
        return;
    }

    if (prev.size() != 1) {
        throw std::runtime_error("ReLU backward expects exactly one parent tensor (CUDA).");
    }

    const float* out_grad_p = static_cast<const float*>(out.grad_ptr().get());
    if (!out_grad_p) {
        return;
    }

    Tensor& parent = prev[0];
    if (!parent.requires_grad()) {
        return;
    }

    float* parent_grad_p = static_cast<float*>(parent.grad_ptr().get());
    if (!parent_grad_p) {
        throw std::runtime_error("Gradient pointer for ReLU parent is null (CUDA).");
    }

    const float* parent_data_p = static_cast<const float*>(parent.data_ptr().get());
    if (!parent_data_p) {
        throw std::runtime_error("Data pointer for ReLU parent is null (CUDA).");
    }

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        out_grad_p,
        parent_data_p,
        parent_grad_p,
        num_elements
    );

    CUDA_CHECK(cudaGetLastError());
}

