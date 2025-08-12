#include "tensor.h"
#include "autograd/ops.h"

void CpuAutograd::flatten(Tensor& out, std::vector<Tensor>& prev) {
    if (prev.size() != 1) {
        throw std::runtime_error("Flatten backward expects exactly one parent tensor.");
    }

    Tensor& a = prev[0];

    std::shared_ptr<void> out_grad_ptr = out.grad_ptr();
    
    std::shared_ptr<void> a_grad_ptr = a.grad_ptr();

    if (out_grad_ptr == nullptr) {
        return;
    }
    if (a_grad_ptr == nullptr) {
        throw std::runtime_error("Cannot propagate gradient to uninitialized parent gradient in flatten backward.");
    }
    
    if (a.numel() != out.numel()) {
        throw std::logic_error("Mismatch in number of elements between forward and backward tensors in flatten.");
    }

    float* a_grad = static_cast<float*>(a_grad_ptr.get());
    const float* out_grad = static_cast<const float*>(out_grad_ptr.get());

    for (size_t i = 0; i < a.numel(); ++i) {
        a_grad[i] += out_grad[i];
    }
}

