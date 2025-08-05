#include "tensor.h"
#include "autograd/ops.h"
#include <stdexcept>
#include <vector>
#include <cmath>

#define PARALLEL_THRESHOLD 2048

void CpuAutograd::pow(const Tensor& out, std::vector<Tensor>& prev) {
    Tensor t = out;
    const size_t num_elements = out.numel();
    const float* out_grad_p = static_cast<const float*>(t.grad_ptr().get());
    const float* out_data_p = static_cast<const float*>(t.data_ptr().get());

    // Case 1: Tensor ^ Tensor (base and exponent are both tensors)
    if (prev.size() == 2) {
        Tensor& a = prev[0]; // Base
        Tensor& b = prev[1]; // Exponent

        const float* a_data_p = static_cast<const float*>(a.data_ptr().get());
        const float* b_data_p = static_cast<const float*>(b.data_ptr().get());
        float* a_grad_p = static_cast<float*>(a.grad_ptr().get());
        float* b_grad_p = static_cast<float*>(b.grad_ptr().get());

        if (!a_data_p || !b_data_p || !out_data_p || !out_grad_p || !a_grad_p || !b_grad_p) {
            throw std::runtime_error("A data or gradient pointer is null in 'pow' backward pass.");
        }

        const bool a_req_grad = a.requires_grad();
        const bool b_req_grad = b.requires_grad();

        if (a_req_grad && b_req_grad) {
            // Most common case: both gradients are needed.
            #pragma omp parallel for simd schedule(static) if(num_elements > PARALLEL_THRESHOLD)
            for (size_t i = 0; i < num_elements; ++i) {
                const float grad_out = out_grad_p[i];
                const float base = a_data_p[i];
                const float exponent = b_data_p[i];
                // Forward pass should ensure base > 0 for log.
                if (base > 0) {
                    a_grad_p[i] += grad_out * exponent * std::pow(base, exponent - 1.0f);
                    b_grad_p[i] += grad_out * out_data_p[i] * std::log(base);
                }
            }
        } else if (a_req_grad) {
            #pragma omp parallel for simd schedule(static) if(num_elements > PARALLEL_THRESHOLD)
            for (size_t i = 0; i < num_elements; ++i) {
                a_grad_p[i] += out_grad_p[i] * b_data_p[i] * std::pow(a_data_p[i], b_data_p[i] - 1.0f);
            }
        } else if (b_req_grad) {
            #pragma omp parallel for simd schedule(static) if(num_elements > PARALLEL_THRESHOLD)
            for (size_t i = 0; i < num_elements; ++i) {
                const float base = a_data_p[i];
                if (base > 0) {
                    b_grad_p[i] += out_grad_p[i] * out_data_p[i] * std::log(base);
                }
            }
        }
    }
    // Case 2: Tensor ^ Scalar (only the base is a tensor in `prev`)
    else if (prev.size() == 1) {
        Tensor& a = prev[0]; // Base
        if (a.requires_grad()) {
            float* a_grad_p = static_cast<float*>(a.grad_ptr().get());
            const float* a_data_p = static_cast<const float*>(a.data_ptr().get());
            if (!a_data_p || !out_data_p || !out_grad_p || !a_grad_p) {
                 throw std::runtime_error("A data or gradient pointer is null in scalar 'pow' backward pass.");
            }

            // The scalar exponent is not passed, so we must recover it from the forward operation:
            // out = a^k => log(out) = k * log(a) => k = log(out) / log(a)
            // This is potentially unstable; a better framework design would save 'k' in the forward pass.
            // We use the first element and add safety checks.
            const float base_0 = a_data_p[0];
            if (base_0 <= 0) throw std::runtime_error("Log is undefined for non-positive base in 'pow' backward.");
            if (base_0 == 1.0f) throw std::runtime_error("Cannot recover exponent when base is 1 in 'pow' backward.");

            const float scalar_exponent = std::log(out_data_p[0]) / std::log(base_0);
            const float scalar_exp_minus_1 = scalar_exponent - 1.0f;

            #pragma omp parallel for simd schedule(static) if(num_elements > PARALLEL_THRESHOLD)
            for (size_t i = 0; i < num_elements; ++i) {
                a_grad_p[i] += out_grad_p[i] * scalar_exponent * std::pow(a_data_p[i], scalar_exp_minus_1);
            }
        }
    } else {
        throw std::invalid_argument("Invalid number of previous tensors for 'pow' backward pass. Expected 1 or 2.");
    }
}
