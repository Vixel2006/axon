#include "autograd/ops.h"
#include "tensor.h"
#include <stdexcept>

#pragma omp declare simd
void CpuAutograd::add(const Tensor& out, std::vector<Tensor>& prev) {
    Tensor t = out;
    Tensor& a = prev[0];
    Tensor& b = prev[1];
    size_t num_elements = a.numel();

    const float* out_grad_p = static_cast<const float*>(t.grad_ptr().get());
    float* a_grad_p = static_cast<float*>(a.grad_ptr().get());
    float* b_grad_p = static_cast<float*>(b.grad_ptr().get());

    if (!out_grad_p || !a_grad_p || !b_grad_p) {
        throw std::runtime_error("A gradient pointer is null in 'add' backward pass.");
    }

    #pragma omp parallel for simd schedule(static) aligned(out_grad_p, a_grad_p: 64) if(num_elements > 1000)
    for (size_t i = 0; i < num_elements; ++i) {
        if (a.requires_grad()) {
            a_grad_p[i] += out_grad_p[i];
        }
    }

    #pragma omp parallel for simd schedule(static) aligned(out_grad_p, b_grad_p: 64) if(num_elements > 1000)
    for (size_t i = 0; i < num_elements; ++i) {
        if (b.requires_grad()) {
            b_grad_p[i] += out_grad_p[i];
        }
    }
}
