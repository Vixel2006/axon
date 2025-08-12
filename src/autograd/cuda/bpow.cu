#include "autograd/ops.h"
#include "tensor.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void bpow_tt_kernel(const float* out_grad_p, const float* out_p,
                               const float* a_p, const float* b_p,
                               float* a_grad_p, float* b_grad_p,
                               bool a_req_grad, bool b_req_grad,
                               size_t num_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (size_t i = index; i < num_elements; i += stride) {
        const float grad = out_grad_p[i];
        const float out_val = out_p[i];
        const float a_val = a_p[i];
        const float b_val = b_p[i];

        if (a_req_grad) {
            a_grad_p[i] += grad * b_val * out_val / (a_val + 1e-9f);
        }

        if (b_req_grad) {
            if (a_val > 0) {
                 b_grad_p[i] += grad * out_val * logf(a_val);
            }
        }
    }
}

__global__ void bpow_ts_kernel(const float* out_grad_p, const float* out_p,
                               const float* a_p, float* a_grad_p,
                               size_t num_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (size_t i = index; i < num_elements; i += stride) {
        const float grad = out_grad_p[i];
        const float out_val = out_p[i];
        const float a_val = a_p[i];

        if (a_val > 0 && fabsf(a_val) != 1.0f) {
            const float exponent = logf(out_val) / logf(a_val);
            a_grad_p[i] += grad * exponent * out_val / (a_val + 1e-9f);
        }
    }
}

void CudaAutograd::pow(Tensor& out, std::vector<Tensor>& prev) {
    const size_t num_elements = out.numel();
    if (num_elements == 0) {
        return;
    }

    Tensor t = out;

    const float* out_grad_p = static_cast<const float*>(t.grad_ptr().get());
    const float* out_p = static_cast<const float*>(t.data_ptr().get());
    if (!out_grad_p || !out_p) {
        throw std::runtime_error("Output data or gradient pointer is null in 'pow' backward pass (CUDA).");
    }

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    if (prev.size() == 2) {
        Tensor& a = prev[0];
        Tensor& b = prev[1];

        const bool a_req_grad = a.requires_grad();
        const bool b_req_grad = b.requires_grad();

        if (!a_req_grad && !b_req_grad) return;

        const float* a_p = static_cast<const float*>(a.data_ptr().get());
        const float* b_p = static_cast<const float*>(b.data_ptr().get());
        float* a_grad_p = static_cast<float*>(a.grad_ptr().get());
        float* b_grad_p = static_cast<float*>(b.grad_ptr().get());

        if (!a_p || !b_p || (a_req_grad && !a_grad_p) || (b_req_grad && !b_grad_p)) {
            throw std::runtime_error("A data or gradient pointer is null in 'pow' (tensor^tensor) backward pass (CUDA).");
        }

        bpow_tt_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            out_grad_p, out_p, a_p, b_p, a_grad_p, b_grad_p,
            a_req_grad, b_req_grad, num_elements);

    } else if (prev.size() == 1) {
        Tensor& a = prev[0];
        if (!a.requires_grad()) return;

        const float* a_p = static_cast<const float*>(a.data_ptr().get());
        float* a_grad_p = static_cast<float*>(a.grad_ptr().get());

        if (!a_p || !a_grad_p) {
             throw std::runtime_error("A data or gradient pointer is null in 'pow' (tensor^scalar) backward pass (CUDA).");
        }

        bpow_ts_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            out_grad_p, out_p, a_p, a_grad_p, num_elements);

    } else {
        throw std::runtime_error("Invalid number of previous tensors for 'pow' backward pass.");
    }

    CUDA_CHECK(cudaGetLastError());
}
