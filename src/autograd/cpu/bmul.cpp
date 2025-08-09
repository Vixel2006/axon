#include "autograd/ops.h"
#include "tensor.h"
#include <stdexcept>

#define PARALLEL_THRESHOLD 4096

void CpuAutograd::mul(const Tensor& out, std::vector<Tensor>& prev) {
    Tensor t = out;
    const size_t num_elements = out.numel();
    const float* out_grad_p = static_cast<const float*>(t.grad_ptr().get());
    if (!out_grad_p) {
        throw std::runtime_error("Output gradient pointer is null in 'mul' backward pass.");
    }

    if (prev.size() == 2) {
        Tensor& a = prev[0];
        Tensor& b = prev[1];

        const float* a_data_p = static_cast<const float*>(a.data_ptr().get());
        const float* b_data_p = static_cast<const float*>(b.data_ptr().get());
        float* a_grad_p = static_cast<float*>(a.grad_ptr().get());
        float* b_grad_p = static_cast<float*>(b.grad_ptr().get());

        const bool a_req_grad = a.requires_grad();
        const bool b_req_grad = b.requires_grad();

        if (!a_req_grad && !b_req_grad) return;

        if ((a_req_grad && (!a_grad_p || !b_data_p)) || (b_req_grad && (!b_grad_p || !a_data_p))) {
            throw std::runtime_error("A data or gradient pointer is null in tensor 'mul' backward pass.");
        }

        if (a_req_grad && b_req_grad) {
            #pragma omp parallel for simd schedule(static) if(num_elements > PARALLEL_THRESHOLD)
            for (size_t i = 0; i < num_elements; ++i) {
                const float grad_out = out_grad_p[i];
                a_grad_p[i] += grad_out * b_data_p[i];
                b_grad_p[i] += grad_out * a_data_p[i];
            }
        } else if (a_req_grad) {
            #pragma omp parallel for simd schedule(static) if(num_elements > PARALLEL_THRESHOLD)
            for (size_t i = 0; i < num_elements; ++i) {
                a_grad_p[i] += out_grad_p[i] * b_data_p[i];
            }
        } else if (b_req_grad) {
            #pragma omp parallel for simd schedule(static) if(num_elements > PARALLEL_THRESHOLD)
            for (size_t i = 0; i < num_elements; ++i) {
                b_grad_p[i] += out_grad_p[i] * a_data_p[i];
            }
        }
    }
    else if (prev.size() == 1) {
        Tensor& a = prev[0];

        if (!a.requires_grad()) return;

        const float* a_data_p = static_cast<const float*>(a.data_ptr().get());
        const float* out_data_p = static_cast<const float*>(out.data_ptr().get());
        float* a_grad_p = static_cast<float*>(a.grad_ptr().get());

        if (!a_data_p || !out_data_p || !a_grad_p) {
            throw std::runtime_error("A data or gradient pointer is null in scalar 'mul' backward pass.");
        }

        #pragma omp parallel for simd schedule(static) if(num_elements > PARALLEL_THRESHOLD)
        for (size_t i = 0; i < num_elements; ++i) {
            const float a_val = a_data_p[i];
            if (a_val != 0.0f) {
                const float scalar_operand = out_data_p[i] / a_val;
                a_grad_p[i] += out_grad_p[i] * scalar_operand;
            }
        }
    }
    else {
        throw std::runtime_error("Invalid number of inputs for mul backward pass.");
    }
}
