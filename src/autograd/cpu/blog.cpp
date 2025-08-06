#include "tensor.h"
#include "autograd/ops.h"
#include <stdexcept>
#include <vector>

#define PARALLEL_THRESHOLD 4096

void CpuAutograd::log(const Tensor& out, std::vector<Tensor>& prev) {
    if (prev.size() != 1) {
        throw std::invalid_argument("Invalid number of previous tensors for 'log' backward pass. Expected 1.");
    }

    Tensor t = out;
    Tensor& a = prev[0];

    if (!a.requires_grad()) {
        return;
    }

    const size_t num_elements = a.numel();
    const float* a_data_p = static_cast<const float*>(a.data_ptr().get());
    const float* out_grad_p = static_cast<const float*>(t.grad_ptr().get());
    float* a_grad_p = static_cast<float*>(a.grad_ptr().get());

    if (!a_data_p || !out_grad_p || !a_grad_p) {
        throw std::runtime_error("A data or gradient pointer is null in 'log' backward pass.");
    }

    #pragma omp parallel for simd schedule(static) if(num_elements > PARALLEL_THRESHOLD)
    for (size_t i = 0; i < num_elements; ++i) {
        a_grad_p[i] += out_grad_p[i] / a_data_p[i];
    }
}
