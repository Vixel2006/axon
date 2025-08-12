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

void CudaAutograd::add(Tensor& out, std::vector<Tensor>& prev) {
    Tensor t = out;
    const size_t num_elements = out.numel();
    if (num_elements == 0) {
        return;
    }

    const float* out_grad_p = static_cast<const float*>(t.grad_ptr().get());
    if (!out_grad_p) {
        throw std::runtime_error("Output gradient pointer is null in 'add' backward pass (CUDA).");
    }

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    if (prev.size() == 2) {
        Tensor& a = prev[0];
        Tensor& b = prev[1];
        
        const bool a_req_grad = a.requires_grad();
        const bool b_req_grad = b.requires_grad();

        if (!a_req_grad && !b_req_grad) {
            return;
        }

        float* a_grad_p = a_req_grad ? static_cast<float*>(a.grad_ptr().get()) : nullptr;
        float* b_grad_p = b_req_grad ? static_cast<float*>(b.grad_ptr().get()) : nullptr;

        if (a_req_grad && !a_grad_p) throw std::runtime_error("Gradient pointer for 'a' is null.");
        if (b_req_grad && !b_grad_p) throw std::runtime_error("Gradient pointer for 'b' is null.");

        badd_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            out_grad_p,
            a_grad_p,
            b_grad_p,
            a_req_grad,
            b_req_grad,
            num_elements
        );
    }
    else if (prev.size() == 1) {
        Tensor& a = prev[0];
        const bool a_req_grad = a.requires_grad();

        if (!a_req_grad) {
            return;
        }
        
        float* a_grad_p = static_cast<float*>(a.grad_ptr().get());
        if (!a_grad_p) {
            throw std::runtime_error("Gradient pointer for 'a' is null in scalar add backward pass.");
        }

        badd_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            out_grad_p,
            a_grad_p,
            nullptr,
            a_req_grad,
            false,
            num_elements
        );
    }
    else {
        throw std::runtime_error("Invalid number of inputs for add backward pass (CUDA).");
    }

    CUDA_CHECK(cudaGetLastError());
}
