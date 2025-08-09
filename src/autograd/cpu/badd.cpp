#include "autograd/ops.h"
#include "tensor.h"
#include <stdexcept>
#include <numeric>

#pragma omp declare simd
void CpuAutograd::add(const Tensor& out, std::vector<Tensor>& prev) {
    Tensor t = out;
    const float* out_grad_p = static_cast<const float*>(t.grad_ptr().get());
    if (!out_grad_p) {
        throw std::runtime_error("Output gradient pointer is null in 'add' backward pass.");
    }

    size_t num_elements = out.numel();

    if (prev.size() == 2) {
        Tensor& a = prev[0];
        Tensor& b = prev[1];

        if (a.requires_grad()) {
            float* a_grad_p = static_cast<float*>(a.grad_ptr().get());
            if (!a_grad_p) throw std::runtime_error("Gradient pointer for 'a' is null.");

            #pragma omp parallel for simd schedule(static) if(num_elements > 1000)
            for (size_t i = 0; i < num_elements; ++i) {
                a_grad_p[i] += out_grad_p[i];
            }
        }

        // Accumulate gradient for the second tensor 'b'
        if (b.requires_grad()) {
            float* b_grad_p = static_cast<float*>(b.grad_ptr().get());
            if (!b_grad_p) throw std::runtime_error("Gradient pointer for 'b' is null.");

            #pragma omp parallel for simd schedule(static) if(num_elements > 1000)
            for (size_t i = 0; i < num_elements; ++i) {
                b_grad_p[i] += out_grad_p[i];
            }
        }
    }
    else if (prev.size() == 1) {
        Tensor& a = prev[0];

        if (a.requires_grad()) {
            float* a_grad_p = static_cast<float*>(a.grad_ptr().get());
            if (!a_grad_p) throw std::runtime_error("Gradient pointer for 'a' is null.");

            #pragma omp parallel for simd schedule(static) if(num_elements > 1000)
            for (size_t i = 0; i < num_elements; ++i) {
                a_grad_p[i] += out_grad_p[i];
            }
        }
    }
    else {
        throw std::runtime_error("Invalid number of inputs for add backward pass.");
    }
}
