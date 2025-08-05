#include "autograd/ops.h"
#include "tensor.h"
#include <stdexcept>

// It is beneficial to define a threshold for parallelization.
// The overhead of creating threads is not worth it for small tensors.
// This value may need tuning based on the specific hardware.
#define PARALLEL_THRESHOLD 4096

void CpuAutograd::mul(const Tensor& out, std::vector<Tensor>& prev) {
    Tensor t = out;
    Tensor& a = prev[0];
    Tensor& b = prev[1];
    const size_t num_elements = a.numel();

    const float* a_data_p = static_cast<const float*>(a.data_ptr().get());
    const float* b_data_p = static_cast<const float*>(b.data_ptr().get());
    const float* out_grad_p = static_cast<const float*>(t.grad_ptr().get());
    float* a_grad_p = static_cast<float*>(a.grad_ptr().get());
    float* b_grad_p = static_cast<float*>(b.grad_ptr().get());

    if (!a_data_p || !b_data_p || !out_grad_p || !a_grad_p || !b_grad_p) {
        throw std::runtime_error("A data or gradient pointer is null in 'mul' backward pass.");
    }

    const bool a_req_grad = a.requires_grad();
    const bool b_req_grad = b.requires_grad();

    if (a_req_grad && b_req_grad) {
        // Most common case: both gradients are needed.
        // Combine into a single loop to reduce overhead and improve cache usage.
        #pragma omp parallel for simd schedule(static) if(num_elements > PARALLEL_THRESHOLD)
        for (size_t i = 0; i < num_elements; ++i) {
            const float grad_out = out_grad_p[i];
            a_grad_p[i] += grad_out * b_data_p[i];
            b_grad_p[i] += grad_out * a_data_p[i];
        }
    } else if (a_req_grad) {
        // Only 'a' needs gradient.
        #pragma omp parallel for simd schedule(static) if(num_elements > PARALLEL_THRESHOLD)
        for (size_t i = 0; i < num_elements; ++i) {
            a_grad_p[i] += out_grad_p[i] * b_data_p[i];
        }
    } else if (b_req_grad) {
        // Only 'b' needs gradient.
        #pragma omp parallel for simd schedule(static) if(num_elements > PARALLEL_THRESHOLD)
        for (size_t i = 0; i < num_elements; ++i) {
            b_grad_p[i] += out_grad_p[i] * a_data_p[i];
        }
    }
}
