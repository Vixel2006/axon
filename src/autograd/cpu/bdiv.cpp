#include "tensor.h"
#include "autograd/ops.h"
#include <stdexcept>
#include <vector>

#define PARALLEL_THRESHOLD 4096

void CpuAutograd::div(Tensor& out, std::vector<Tensor>& prev) {
    Tensor t = out;
    const size_t num_elements = out.numel();
    const float* out_grad_p = static_cast<const float*>(t.grad_ptr().get());

    if (prev.size() == 2) {
        Tensor& a = prev[0];
        Tensor& b = prev[1];

        const float* a_data_p = static_cast<const float*>(a.data_ptr().get());
        const float* b_data_p = static_cast<const float*>(b.data_ptr().get());
        float* a_grad_p = static_cast<float*>(a.grad_ptr().get());
        float* b_grad_p = static_cast<float*>(b.grad_ptr().get());

        if (!a_data_p || !b_data_p || !out_grad_p || !a_grad_p || !b_grad_p) {
            throw std::runtime_error("A data or gradient pointer is null in 'div' backward pass.");
        }

        const bool a_req_grad = a.requires_grad();
        const bool b_req_grad = b.requires_grad();

        if (a_req_grad && b_req_grad) {
            #pragma omp parallel for simd schedule(static) if(num_elements > PARALLEL_THRESHOLD)
            for (size_t i = 0; i < num_elements; ++i) {
                const float b_inv = 1.0f / b_data_p[i];
                const float grad_out = out_grad_p[i];
                a_grad_p[i] += grad_out * b_inv;
                b_grad_p[i] -= grad_out * a_data_p[i] * b_inv * b_inv;
            }
        } else if (a_req_grad) {
            #pragma omp parallel for simd schedule(static) if(num_elements > PARALLEL_THRESHOLD)
            for (size_t i = 0; i < num_elements; ++i) {
                a_grad_p[i] += out_grad_p[i] / b_data_p[i];
            }
        } else if (b_req_grad) {
            #pragma omp parallel for simd schedule(static) if(num_elements > PARALLEL_THRESHOLD)
            for (size_t i = 0; i < num_elements; ++i) {
                const float b_inv = 1.0f / b_data_p[i];
                b_grad_p[i] -= out_grad_p[i] * a_data_p[i] * b_inv * b_inv;
            }
        }
    }
    else if (prev.size() == 1) {
        Tensor& a = prev[0];

        if (a.requires_grad()) {
            const float* a_data_p = static_cast<const float*>(a.data_ptr().get());
            const float* out_data_p = static_cast<const float*>(t.data_ptr().get());
            float* a_grad_p = static_cast<float*>(a.grad_ptr().get());

            if (!a_data_p || !out_data_p || !out_grad_p || !a_grad_p) {
                throw std::runtime_error("A data or gradient pointer is null in scalar 'div' backward pass.");
            }

            if (out_data_p[0] == 0.0f) {
                throw std::runtime_error("Division by zero while recovering scalar in 'div' backward pass.");
            }
            const float scalar = a_data_p[0] / out_data_p[0];
            const float inv_scalar = 1.0f / scalar;

            #pragma omp parallel for simd schedule(static) if(num_elements > PARALLEL_THRESHOLD)
            for (size_t i = 0; i < num_elements; ++i) {
                a_grad_p[i] += out_grad_p[i] * inv_scalar;
            }
        }
    } else {
        throw std::invalid_argument("Invalid number of previous tensors for 'div' backward pass. Expected 1 or 2.");
    }
}
