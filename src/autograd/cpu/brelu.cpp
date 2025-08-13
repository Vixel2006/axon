#include "autograd/ops.h"
#include "tensor.h"
#include <stdexcept>

#pragma omp declare simd
void CpuAutograd::relu(Tensor& out, std::vector<Tensor>& prev) {
    if (prev.size() != 1) {
        throw std::runtime_error("ReLU backward expects exactly one parent tensor.");
    }
    
    const float* out_grad_p = static_cast<const float*>(out.grad_ptr().get());
    if (out_grad_p == nullptr) {
        return;
    }

    Tensor& parent = prev[0];
    if (!parent.requires_grad()) {
        return;
    }
    
    float* parent_grad_p = static_cast<float*>(parent.grad_ptr().get());
    if (parent_grad_p == nullptr) {
        throw std::runtime_error("Gradient pointer for ReLU parent is null. Did you forget to call zero_grad()?");
    }

    const float* parent_data_p = static_cast<const float*>(parent.data_ptr().get());
    if (parent_data_p == nullptr) {
        throw std::runtime_error("Data pointer for ReLU parent is null.");
    }

    size_t num_elements = parent.numel();


    #pragma omp parallel for simd schedule(static) if(num_elements > 1000)
    for (size_t i = 0; i < num_elements; ++i) {
        if (parent_data_p[i] > 0) {
            parent_grad_p[i] += out_grad_p[i];
        }
    }
}
