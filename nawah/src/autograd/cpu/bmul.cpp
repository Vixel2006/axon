#include "autograd/ops.h"
#include "tensor.h"

void CpuAutograd::mul(const Tensor& out, std::vector<Tensor>& prev) {
    Tensor t = out;
    Tensor& a = prev[0];
    Tensor& b = prev[1];
    size_t num_elements = a.numel();

    const float* a_data_p = static_cast<const float*>(a.data_ptr().get());
    const float* b_data_p = static_cast<const float*>(b.data_ptr().get());

    const float* out_grad_p = static_cast<float*>(t.grad_ptr().get());
    float* a_grad_p = static_cast<float*>(a.grad_ptr().get());
    float* b_grad_p = static_cast<float*>(b.grad_ptr().get());

    if (!a_data_p || !b_data_p || !out_grad_p || !a_grad_p || !b_grad_p) {
        throw std::runtime_error("A data or gradient pointer is null in backward pass.");
    }
    
    if (a.requires_grad()) {
        for (size_t i = 0; i < num_elements; ++i) {
            a_grad_p[i] += out_grad_p[i] * b_data_p[i];
        }
    }

    if (b.requires_grad()) {
        for (size_t i = 0; i < num_elements; ++i) {
            b_grad_p[i] += out_grad_p[i] * a_data_p[i];
        }
    }
}
